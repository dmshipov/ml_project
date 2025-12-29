from pathlib import Path
from typing import List, Optional
import json
import hashlib
import random
import logging
import os

import numpy as np
import onnxruntime as ort
import redis
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator


# Расположения файлов моделей
MODEL_PATH_MAIN = Path("models/default_model_int.onnx")   # главная (оптимизированная)
MODEL_PATH_AB = Path("models/default_model.onnx")          # дополнительная для A/B эксперимента

logger = logging.getLogger(__name__)


class Features(BaseModel):
    """Единичный элемент с характеристиками."""
    values: List[float]


class BatchRequest(BaseModel):
    """Группа элементов для выполнения предсказаний."""
    batch: List[Features]


app = FastAPI(title="Credit scoring API")

# Инструменты мониторинга Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app, include_in_schema=False)

# --- Настройка Redis -------------------------------------------------------------

# Конфигурация Redis (сервис redis из docker-compose)
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = 0

redis_client: Optional[redis.Redis] = None

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )
    # Проверяем соединение пингом, чтобы избежать сбоев от неработающего Redis
    redis_client.ping()
    logger.info("Redis доступен на %s:%s", REDIS_HOST, REDIS_PORT)
except Exception as exc:  # noqa: BLE001
    logger.warning(
        "Redis недоступен (%s). Функция кеширования будет выключена.", exc
    )
    redis_client = None


def make_cache_key(prefix: str, request: BatchRequest) -> str:
    """
    Создаём ключ для кеша, основываясь на данных запроса.
    prefix — обозначение версии модели (к примеру, main или ab-A).
    """
    payload = [row.values for row in request.batch]
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"credit:{prefix}:{digest}"


# --- Инициализация моделей ONNX ---------------------------------------------


def create_session(path: Path) -> ort.InferenceSession:
    """Упрощённая функция для запуска сессии ONNX Runtime."""
    return ort.InferenceSession(
        path.as_posix(),
        providers=["CPUExecutionProvider"],
    )


# главная оптимизированная модель (применяется в /predict и варианте B)
session_main = create_session(MODEL_PATH_MAIN)
input_name_main = session_main.get_inputs()[0].name
output_name_main = session_main.get_outputs()[0].name

# альтернативная модель для A/B эксперимента (вариант A)
session_ab = create_session(MODEL_PATH_AB)
input_name_ab = session_ab.get_inputs()[0].name
output_name_ab = session_ab.get_outputs()[0].name


@app.on_event("startup")
def startup_event() -> None:
    """
    Обработчик запуска приложения.
    В настоящее время инициализация происходит на уровне модуля,
    поэтому это лишь заглушка.
    """
    logger.info("Сервис кредитного скоринга API успешно стартовал.")


@app.get("/health")
def health() -> dict:
    """Базовый маршрут для проверки состояния контейнера."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: BatchRequest) -> dict:
    """
    Обычное предсказание без кеширования и A/B тестирования.
    Применяет главную оптимизированную модель.
    """
    matrix = np.array(
        [row.values for row in request.batch],
        dtype="float32",
    )
    raw = session_main.run([output_name_main], {input_name_main: matrix})[0]
    preds = raw.tolist()
    return {"predictions": preds}


@app.post("/predict_cached")
def predict_cached(request: BatchRequest) -> dict:
    """
    Предсказание с сохранением результатов в Redis.
    При недоступности Redis работает как стандартный /predict без ошибок.
    """
    matrix = np.array(
        [row.values for row in request.batch],
        dtype="float32",
    )

    # Если Redis не настроен — просто выполняем расчёт и возвращаем результат.
    if redis_client is None:
        raw = session_main.run([output_name_main], {input_name_main: matrix})[0]
        preds = raw.tolist()
        return {
            "predictions": preds,
            "cached": False,
        }

    cache_key = make_cache_key(prefix="main", request=request)

    # Пробуем извлечь данные из кеша, но не позволяем Redis прервать работу API.
    try:
        cached = redis_client.get(cache_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Проблема при извлечении данных из Redis: %s", exc)
        cached = None

    if cached is not None:
        try:
            preds = json.loads(cached)
            return {
                "predictions": preds,
                "cached": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка при декодировании кеша: %s", exc)

    # Кеш отсутствует или повреждён — пересчитываем.
    raw = session_main.run([output_name_main], {input_name_main: matrix})[0]
    preds = raw.tolist()

    # Сохраняем в кеш, но снова не даём Redis нарушить ответ.
    try:
        redis_client.set(cache_key, json.dumps(preds, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не получилось сохранить результат в Redis: %s", exc)

    return {
        "predictions": preds,
        "cached": False,
    }


@app.post("/predict_ab")
def predict_ab(request: BatchRequest) -> dict:
    """
    A/B эксперимент: некоторые запросы обрабатываются главной моделью,
    другие — альтернативной. Выбор версии происходит случайно.
    """
    # Упрощённый подход: равное распределение 50% / 50%
    variant = "A" if random.random() < 0.5 else "B"

    matrix = np.array(
        [row.values for row in request.batch],
        dtype="float32",
    )

    if variant == "A":
        # Версия A — базовая модель ONNX
        raw = session_ab.run([output_name_ab], {input_name_ab: matrix})[0]
    else:
        # Версия B — оптимизированная модель (аналогично /predict)
        raw = session_main.run([output_name_main], {input_name_main: matrix})[0]

    preds = raw.tolist()

    return {
        "variant": variant,
        "predictions": preds,
    }
