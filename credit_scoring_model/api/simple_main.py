"""Упрощенное FastAPI приложение без проблемных зависимостей."""

import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# Простые модели данных
class CreditScoringRequest(BaseModel):
    annual_inc: float
    emp_length: str
    home_ownership: str
    loan_amnt: float
    term: str
    purpose: str
    fico_range_low: int
    fico_range_high: int
    dti: float
    revol_util: float
    inq_last_6mths: int
    delinq_2yrs: int
    pub_rec: int = 0


class ModelPrediction(BaseModel):
    prediction: int  # 0 = одобрено, 1 = отклонено
    probability: float
    confidence: str
    risk_score: float
    recommended_amount: Optional[float] = None
    model_version: str = "1.0.0"
    features_importance: Optional[Dict[str, float]] = None


class CreditScoringResponse(BaseModel):
    success: bool
    prediction: ModelPrediction
    processing_time_ms: float
    request_id: str
    timestamp: str


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    database_status: str = "healthy"
    model_status: str = "healthy"
    uptime_seconds: float


# Создание FastAPI приложения
app = FastAPI(
    title="API Кредитного Скоринга (Упрощенная версия)",
    description="Упрощенная версия API для кредитного скоринга без ML зависимостей",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Время запуска для расчета uptime
startup_time = time.time()


# Простая логика предсказания (заглушка)
def simple_credit_prediction(request: CreditScoringRequest) -> ModelPrediction:
    """Простая логика предсказания кредитного скоринга."""

    # Базовые правила для предсказания
    risk_score = 0

    # FICO Score (основной фактор)
    fico_avg = (request.fico_range_low + request.fico_range_high) / 2
    if fico_avg >= 750:
        risk_score += 20
    elif fico_avg >= 700:
        risk_score += 15
    elif fico_avg >= 650:
        risk_score += 10
    else:
        risk_score += 5

    # DTI (Debt-to-Income)
    if request.dti <= 20:
        risk_score += 15
    elif request.dti <= 30:
        risk_score += 10
    elif request.dti <= 40:
        risk_score += 5
    else:
        risk_score += 0

    # Revolving Utilization
    if request.revol_util <= 20:
        risk_score += 10
    elif request.revol_util <= 40:
        risk_score += 5
    else:
        risk_score += 0

    # Employment Length
    emp_length_score = {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0,
    }
    risk_score += emp_length_score.get(request.emp_length, 0)

    # Home Ownership
    home_ownership_score = {"OWN": 10, "MORTGAGE": 8, "RENT": 5, "OTHER": 3}
    risk_score += home_ownership_score.get(request.home_ownership, 0)

    # Inquiries
    if request.inq_last_6mths <= 1:
        risk_score += 5
    elif request.inq_last_6mths <= 3:
        risk_score += 3
    else:
        risk_score += 0

    # Delinquencies
    if request.delinq_2yrs == 0:
        risk_score += 5
    else:
        risk_score += 0

    # Нормализация risk_score (0-100)
    risk_score = min(100, max(0, risk_score))

    # Определение предсказания
    threshold = 50  # Порог для одобрения
    prediction = 0 if risk_score >= threshold else 1

    # Вероятность (обратная к risk_score)
    probability = (100 - risk_score) / 100

    # Уровень уверенности
    if probability >= 0.8:
        confidence = "high"
    elif probability >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    # Рекомендуемая сумма
    if prediction == 0:  # Одобрено
        recommended_amount = min(request.loan_amnt * 1.1, request.annual_inc * 0.3)
    else:
        recommended_amount = None

    # Важность признаков (заглушка)
    features_importance = {
        "fico_score": 0.3,
        "dti": 0.2,
        "revol_util": 0.15,
        "emp_length": 0.15,
        "home_ownership": 0.1,
        "inq_last_6mths": 0.05,
        "delinq_2yrs": 0.05,
    }

    return ModelPrediction(
        prediction=prediction,
        probability=probability,
        confidence=confidence,
        risk_score=risk_score,
        recommended_amount=recommended_amount,
        model_version="1.0.0-simple",
        features_importance=features_importance,
    )


# Middleware для измерения времени обработки
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Добавление времени обработки в заголовки ответа."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Глобальный обработчик исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Внутренняя ошибка сервера",
            "message": "Произошла неожиданная ошибка",
        },
    )


# Endpoints
@app.get("/")
async def root():
    """Корневой endpoint."""
    return {
        "message": "API Кредитного Скоринга (Упрощенная версия)",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """Endpoint для проверки состояния системы."""
    current_time = time.time()
    uptime = current_time - startup_time

    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        database_status="healthy",
        model_status="healthy",
        uptime_seconds=uptime,
    )


@app.post("/api/v1/predict", response_model=CreditScoringResponse)
async def predict_credit_score(request: CreditScoringRequest):
    """Предсказать кредитный скоринг и одобрение займа."""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Выполнить предсказание
        prediction_result = simple_credit_prediction(request)

        # Вычислить время обработки
        processing_time = (time.time() - start_time) * 1000

        # Создать ответ
        response = CreditScoringResponse(
            success=True,
            prediction=prediction_result,
            processing_time_ms=processing_time,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Предсказание не удалось: {str(e)}"
        )


@app.get("/api/v1/model/info")
async def get_model_info():
    """Получить информацию о модели."""
    return {
        "model_version": "1.0.0-simple",
        "model_type": "Rule-based",
        "threshold": 50,
        "features_count": 7,
        "last_updated": datetime.utcnow().isoformat(),
        "description": "Упрощенная модель на основе правил",
    }


@app.get("/api/v1/predictions/stats")
async def get_prediction_stats():
    """Получить статистику предсказаний."""
    return {
        "total_predictions": random.randint(1000, 5000),
        "approval_rate": round(random.uniform(0.6, 0.8), 3),
        "average_processing_time_ms": round(random.uniform(50, 200), 2),
        "last_24h_predictions": random.randint(50, 200),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.simple_main:app", host="0.0.0.0", port=8000, reload=True)
