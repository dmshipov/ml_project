"""
Телеграм-бот для анализа и прогнозирования акций.
Функции:
- загрузка котировок (yfinance, 2 года, колонка 'Close');
- обучение 3 моделей: Ridge(лаги с подбором alpha), ARIMA (с auto_arima), LSTM (с улучшенной архитектурой);
- сравнение по RMSE и MAPE на тестовом хвосте;
- прогноз на 30 дней, график (история + прогноз);
- рекомендации по локальным минимумам/максимумам;
- оценка условной прибыли на введённую сумму;
- логирование в logs.csv.

Зависимости: aiogram, yfinance, numpy, pandas, scikit-learn, statsmodels, matplotlib, tensorflow (для LSTM), pmdarima (для auto_arima).
Если TensorFlow недоступен, LSTM пропускается.
Если pmdarima недоступен, ARIMA использует фиксированные параметры.
"""

import os
import io
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import InlineKeyboardBuilder

import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Статистика
from statsmodels.tsa.arima.model import ARIMA

# Попытка импорта pmdarima для auto_arima
PMDARIMA_AVAILABLE = True
try:
    from pmdarima import auto_arima
except Exception:
    PMDARIMA_AVAILABLE = False

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    TF_AVAILABLE = False

# Базовая настройка логера (для консоли)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-bot")

# Список популярных тикеров
TOP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "BABA", "TSM",
    "V", "JPM", "WMT", "DIS", "KO", "ORCL", "CRM", "AMD", "INTC", "IBM"
]

# Состояния пользователей для выбора тикера и суммы
user_states: Dict[int, Dict[str, str]] = {}


# ВСПОМОГАТЕЛЬНЫЕ ШТУКИ

@dataclass
class EvalResult:
    name: str
    rmse: float
    mape: float
    model_obj: object
    extra: Dict[str, object]

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE в процентах (добавляем эпсилон)."""
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

def train_test_split_series(series: pd.Series, test_size_days: int = 30) -> Tuple[pd.Series, pd.Series]:
    """Разбиваем временной ряд на train/test по последним дням (по индексу)."""
    if len(series) <= test_size_days + 30:
        # запас для обучения
        test_size_days = max(7, len(series) // 5)
    return series.iloc[:-test_size_days], series.iloc[-test_size_days:]

def remove_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Удаляем выбросы с помощью Z-score."""
    z_scores = np.abs((series - series.mean()) / series.std())
    return series[z_scores < threshold]

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляем простые технические индикаторы: RSI, MACD."""
    # RSI
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['y'].ewm(span=12, adjust=False).mean()
    ema26 = df['y'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    return df

def build_lag_features(y: pd.Series, max_lag: int = 10, ma_windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """
    Принимаем Series/ndarray/DataFrame.
    Превращаем в 1D Series (если DataFrame — берём первый числовой столбец).
    Сохраняем индекс, если он совместим по длине.
    Добавляем технические индикаторы.
    """
    # Если это DataFrame — берём первый числовой столбец
    if isinstance(y, pd.DataFrame):
        num_cols = y.select_dtypes(include=[np.number])
        if num_cols.shape[1] == 0:
            raise ValueError("Входной DataFrame не содержит числовых столбцов.")
        y_series = num_cols.iloc[:, 0]
    else:
        y_series = y

    # Индекс, если доступен
    orig_index = getattr(y_series, "index", None)

    # Приводим к np и сплющиваем
    arr = np.asarray(y_series)
    arr = np.squeeze(arr) 
    if arr.ndim != 1:
        raise ValueError(f"Ожидался одномерный ряд, а пришло {arr.shape}")

    # Собираем обратно Series с индексом, если он валиден
    if orig_index is None or len(orig_index) != len(arr):
        y1d = pd.Series(arr, dtype="float64")
    else:
        y1d = pd.Series(arr, index=orig_index, dtype="float64")

    df = y1d.to_frame(name="y")

    for lag in range(1, max_lag + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for w in ma_windows:
        df[f"ma_{w}"] = df["y"].rolling(w, min_periods=1).mean().shift(1)

    # Добавляем индикаторы
    df = add_technical_indicators(df)

    df = df.dropna()
    return df

def ridge_fit_predict(y: pd.Series, test_len: int) -> EvalResult:
    """Ridge с подбором alpha через GridSearch."""
    feat = build_lag_features(y, max_lag=10, ma_windows=[3, 7, 14])

    # Если после лагов мало строк — уменьшим хвост (но не меньше 7)
    if len(feat) <= test_len:
        test_len = max(7, len(feat) // 5)

    X = feat.drop(columns=["y"])
    y_target = feat["y"]

    X_train, X_test = X.iloc[:-test_len], X.iloc[-test_len:]
    y_train, y_test = y_target.iloc[:-test_len], y_target.iloc[-test_len:]

    # Подбор alpha
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(Ridge(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    preds = model.predict(X_test)

    return EvalResult(
        name="RIDGE",
        rmse=math.sqrt(mean_squared_error(y_test, preds)),
        mape=mape(y_test.values, preds),
        model_obj=(model, X.columns.tolist()),
        extra={"y_test_index": y_test.index, "preds": preds},
    )

def arima_fit_predict(y: pd.Series, test_len: int) -> EvalResult:
    """ARIMA с auto_arima если доступен, иначе фиксированные параметры."""
    train, test = train_test_split_series(y, test_len)
    
    if PMDARIMA_AVAILABLE:
        model = auto_arima(train, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
        fit = model.fit(train)
        name = f"ARIMA{model.order}"
    else:
        order = (1, 1, 1)
        model_arima = ARIMA(train.values, order=order)
        fit = model_arima.fit()
        name = "ARIMA(1,1,1)"

    preds = fit.forecast(steps=len(test))
    preds = np.asarray(preds, dtype=float)

    res = EvalResult(
        name=name,
        rmse=math.sqrt(mean_squared_error(test.values, preds)),
        mape=mape(test.values, preds),
        model_obj=fit,
        extra={"y_test_index": test.index, "preds": preds},
    )
    return res

def lstm_fit_predict(y: pd.Series, test_len: int) -> Optional[EvalResult]:
    if not TF_AVAILABLE:
        return None

    series = y.values.astype("float32")
    train, test = train_test_split_series(y, test_len)
    tr = train.values.astype("float32")
    mn, mx = float(tr.min()), float(tr.max())
    rng = (mx - mn) if (mx - mn) > 1e-8 else 1.0

    norm = (series - mn) / rng
    window = 20

    X_all, y_all = make_lstm_dataset(norm, window)
    X_all = X_all.astype("float32")
    y_all = y_all.astype("float32")

    split_idx = len(train) - window
    if split_idx < 10:
        split_idx = int(max(10, 0.8 * len(X_all)))

    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]

    # Улучшенная архитектура: dropout, early stopping
    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=50,  # Увеличено
        batch_size=16,
        verbose=0,
        validation_split=0.1,
        callbacks=[early_stop],
    )

    y_pred_norm = model.predict(X_test, verbose=0).ravel().astype("float32")
    y_test_denorm = y_test * rng + mn
    y_pred_denorm = y_pred_norm * rng + mn

    return EvalResult(
        name="LSTM",
        rmse=math.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm)),
        mape=mape(y_test_denorm, y_pred_denorm),
        model_obj=(model, mn, mx, rng, window),
        extra={"y_test_index": y.index[-len(y_test_denorm):], "preds": y_pred_denorm},
    )

def pick_best_model(results: List[EvalResult]) -> EvalResult:
    """Выбор по среднему рангу RMSE/MAPE. Меньше — лучше."""
    df = pd.DataFrame([
        {"name": r.name, "rmse": r.rmse, "mape": r.mape}
        for r in results
    ])
    df["rank_rmse"] = df["rmse"].rank(method="min")
    df["rank_mape"] = df["mape"].rank(method="min")
    df["rank_mean"] = (df["rank_rmse"] + df["rank_mape"]) / 2.0
    best_name = df.sort_values("rank_mean").iloc[0]["name"]
    for r in results:
        if r.name == best_name:
            return r
    return results[0]

def ensemble_forecast(results: List[EvalResult], y: pd.Series, horizon: int = 30) -> pd.Series:
    """Ансамбль: усредняем прогнозы всех моделей с весами по обратной MAPE."""
    forecasts = []
    weights = []
    last_date = y.index[-1]
    index = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    
    for res in results:
        fc = forecast_30(res, y)
        forecasts.append(fc.values)
        weights.append(1.0 / (res.mape + 1e-8))  # Вес по обратной MAPE
    
    weights = np.array(weights)
    weights /= weights.sum()  # Нормализация
    
    ensemble_preds = np.average(np.array(forecasts), axis=0, weights=weights)
    return pd.Series(ensemble_preds, index=index)

def forecast_30(best: EvalResult, y: pd.Series) -> pd.Series:
    """Делаем прогноз на 30 дней вперёд в зависимости от типа модели."""
    horizon = 30
    last_date = y.index[-1]

    if best.name == "RIDGE":
        model, cols = best.model_obj
        # Историю приводим к 1D Series
        if isinstance(y, pd.DataFrame):
            y_num = y.select_dtypes(include=[np.number])
            base = y_num.iloc[:, 0]
        else:
            base = pd.Series(np.asarray(y).squeeze(), index=y.index, dtype="float64")

        hist = base.copy()
        preds = []

        for _ in range(horizon):
            feat_full = build_lag_features(hist, max_lag=10, ma_windows=[3, 7, 14])
            if feat_full.empty:
                # если вдруг слишком короткая история — прерываемся
                break
            feat = feat_full.iloc[-1:]
            x_row = feat.drop(columns=["y"]).copy()

            # добиваем отсутствующие признаки нулями
            for c in cols:
                if c not in x_row.columns:
                    x_row[c] = 0.0
            x_row = x_row[cols]

            y_next = float(model.predict(x_row)[0])
            preds.append(y_next)

            # расширяем hist аккуратно
            next_idx = hist.index[-1] + pd.Timedelta(days=1)
            hist = pd.concat([hist, pd.Series([y_next], index=[next_idx], dtype="float64")])

        index = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(preds), freq="D")
        return pd.Series(preds, index=index, dtype="float64")

    if best.name.startswith("ARIMA"):
        train, _ = train_test_split_series(y, 30)
        fit = best.model_obj
        fc = fit.forecast(steps=horizon)
        index = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(np.asarray(fc, dtype=float), index=index)

    if best.name == "LSTM":
        model, mn, mx, rng, window = best.model_obj
        hist = y.values.astype("float32")
        preds = []

        norm_hist = (hist - mn) / (rng if rng > 1e-8 else 1.0)
        norm_hist = norm_hist.astype("float32")

        for _ in range(horizon):
            if len(norm_hist) < window:
                pad = np.zeros(window - len(norm_hist), dtype="float32")
                seq = np.concatenate([pad, norm_hist]).astype("float32")
            else:
                seq = norm_hist[-window:].astype("float32")

            x_np = np.asarray(seq, dtype="float32").reshape(1, window, 1)
            x = tf.convert_to_tensor(x_np, dtype=tf.float32)  # ключевая строка
            y_next_norm = float(model.predict(x, verbose=0)[0][0])
            y_next = y_next_norm * (rng if rng > 1e-8 else 1.0) + mn

            preds.append(y_next)
            norm_hist = np.append(norm_hist, np.float32(y_next_norm)).astype("float32")

        index = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(preds, index=index)

def local_extrema(series: pd.Series, window: int = 3) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Ищем локальные минимумы/максимумы на прогнозе.
    Условие минимума: значение меньше соседей в окне; максимума — больше соседей.
    """
    lows, highs = [], []
    vals = series.values
    for i in range(window, len(vals) - window):
        seg = vals[i - window:i + window + 1]
        center = vals[i]
        if np.all(center <= seg) and np.count_nonzero(center < seg) >= 1:
            lows.append(series.index[i])
        if np.all(center >= seg) and np.count_nonzero(center > seg) >= 1:
            highs.append(series.index[i])
    return lows, highs

def simulate_strategy(history_last: float, forecast: pd.Series, cash: float) -> Tuple[float, List[Tuple[str, str, float, float]]]:
    """
    Простая симуляция:
    - покупка в каждом локальном минимуме по цене прогноза,
    - продажа в ближайшем локальном максимуме (если он позже),
    - торгуем на всю сумму каждый раз, без комиссий.
    Возврат: прибыль и сделки [(date_buy, date_sell, price_buy, price_sell), ...]
    """
    lows, highs = local_extrema(forecast, window=2)
    trades = []
    profit = 0.0
    # Сопоставим каждой покупке ближайший будущий максимум
    for lb in lows:
        h_candidates = [h for h in highs if h > lb]
        if not h_candidates:
            continue
        hs = min(h_candidates)
        p_buy = float(forecast.loc[lb])
        p_sell = float(forecast.loc[hs])
        if p_sell <= p_buy:
            continue
        shares = cash / p_buy
        profit += (p_sell - p_buy) * shares
        trades.append((lb.strftime("%Y-%m-%d"), hs.strftime("%Y-%m-%d"), p_buy, p_sell))
    return profit, trades

def calculate_volatility(series: pd.Series) -> float:
    """Расчёт волатильности как стандартное отклонение процентных изменений."""
    returns = series.pct_change().dropna()
    return returns.std() * 100  # в процентах

def calculate_trend(series: pd.Series) -> str:
    """Простой тренд: сравнение первой и последней цены."""
    if len(series) < 2:
        return "Недостаточно данных"
    first = series.iloc[0]
    last = series.iloc[-1]
    pct = ((last - first) / first) * 100
    if pct > 1:
        return f"Восходящий (+{pct:.2f}%)"
    elif pct < -1:
        return f"Нисходящий ({pct:.2f}%)"
    else:
        return f"Боковой ({pct:.2f}%)"

def plot_history_forecast(hist: pd.Series, forecast: pd.Series, buys: List[pd.Timestamp], sells: List[pd.Timestamp], ticker: str, volatility: float, trend: str, delta_pct: float) -> bytes:
    """Строим график (история + прогноз) и помечаем точки покупок/продаж. Добавлены подписи, средняя линия, процентное изменение."""
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist.values, label="История (Close)", color='blue', linewidth=1.5)
    plt.plot(forecast.index, forecast.values, label="Прогноз (30 д.)", color='orange', linewidth=1.5, linestyle='--')
    
    # Средняя линия по истории
    mean_hist = hist.mean()
    plt.axhline(y=mean_hist, color='green', linestyle=':', label=f'Средняя история: {mean_hist:.2f}')
    
    # Отметим экстремумы
    buy_vals = [forecast.loc[d] for d in buys]
    sell_vals = [forecast.loc[d] for d in sells]
    plt.scatter(buys, buy_vals, marker="^", s=80, color='green', label="Покупать")
    plt.scatter(sells, sell_vals, marker="v", s=80, color='red', label="Продавать")
    
    # Подписи осей и заголовок
    plt.xlabel("Дата")
    plt.ylabel("Цена (USD)")
    plt.title(f"Прогноз цены акций {ticker}\nИзменение: {delta_pct:+.2f}%, Волатильность: {volatility:.2f}%, Тренд: {trend}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return buf.read()

def parse_user_text(text: str) -> Tuple[str, float]:
    """
    Ожидаемый формат: 'GOOGL 20000' или 'MSFT, 15000'
    Возвращаем (тикер UPPER, сумма).
    """
    raw = text.replace(",", " ").replace(";", " ").strip().split()
    if len(raw) < 2:
        raise ValueError("Укажите тикер и сумму в USD, например: GOOGL 20000")
    ticker = raw[0].upper()
    try:
        amount = float(raw[1])
    except Exception:
        raise ValueError("Сумма должна быть числом. Пример: GOOGL 20000")
    if amount <= 0:
        raise ValueError("Сумма должна быть больше нуля.")
    return ticker, amount

def safe_log(row: Dict[str, object], path: str = "logs.csv") -> None:
    """Записываем строку в CSV (создаём файл с заголовком при первом запуске)."""
    cols = [
        "timestamp", "user_id", "ticker", "amount",
        "best_model", "rmse", "mape",
        "delta_pct", "profit_est"
    ]
    exists = os.path.exists(path)
    df = pd.DataFrame([row], columns=cols)
    if exists:
        df.to_csv(path, index=False, mode="a", header=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, mode="w", header=True, encoding="utf-8")

def load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    for line in open(path, "r", encoding="utf-8"):
        if "=" in line and not line.strip().startswith("#"):
            key, val = line.strip().split("=", 1)
            os.environ[key] = val
load_dotenv()


# ТЕЛЕГРАМ-БОТ (Aiogram)

TOKEN = os.getenv("TOKEN")

def make_kb() -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="GOOGL 20000", callback_data="GOOGL 20000")
    kb.button(text="MSFT 15000", callback_data="MSFT 15000")
    kb.button(text="GOOGL 20000", callback_data="GOOGL 20000")
    kb.button(text="TSLA 5000", callback_data="TSLA 5000")
    kb.button(text="AMZN 25000", callback_data="AMZN 25000")
    kb.button(text="Показать список всех тикеров", callback_data="show_tickers")
    kb.adjust(2)  # 2 кнопки в ряд
    return kb.as_markup()

def make_tickers_kb() -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for t in TOP_TICKERS:
        kb.button(text=t, callback_data=f"ticker_{t}")
    kb.button(text="Назад", callback_data="back")
    kb.adjust(3)  # 3 кнопки в ряд
    return kb.as_markup()

async def process_query(text: str, user: types.User) -> None:
    """Основной обработчик: парсинг, загрузка, обучение, ответ пользователю."""
    try:
        ticker, amount = parse_user_text(text)
    except Exception as e:
        await user.bot.send_message(chat_id=user.id, text=f"⛔ {e}", reply_markup=make_kb())
        return

    await user.bot.send_message(chat_id=user.id, text=f"Запрос принят: {ticker}, сумма {amount:.2f}.\nГотовлю данные…")

    # Загрузка котировок
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=365*2)  # 2 года
    logger.info(f"Начинаем загрузку данных для {ticker}: start={start}, end={end}")

    data = None
    try:
        # Попытка 1: yfinance
        data = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            progress=False,
            auto_adjust=False,
            timeout=10
        )
        logger.info(f"yfinance: Данные загружены для {ticker}: shape={data.shape}")
        if data.empty:
            raise ValueError("yfinance вернул пустые данные")
    except Exception as e:
        logger.warning(f"yfinance failed для {ticker}: {e}. Пробуем pandas_datareader...")
        try:
            # Попытка 2: pandas_datareader (альтернатива)
            import pandas_datareader as pdr
            data = pdr.get_data_yahoo(ticker, start=start, end=end)
            logger.info(f"pandas_datareader: Данные загружены для {ticker}: shape={data.shape}")
        except Exception as e2:
            logger.exception(f"pandas_datareader failed для {ticker}: {e2}")
            await user.bot.send_message(chat_id=user.id, text="Не удалось получить котировки. Проверьте тикер сверив его с вложеным списком и попробуйте снова.")
            return

    if data.empty or "Close" not in data.columns:
        logger.warning(f"Данные пустые или нет колонки 'Close' для {ticker}: empty={data.empty}, columns={list(data.columns)}")
        await user.bot.send_message(chat_id=user.id, text="Не удалось получить котировки. Проверьте тикер сверив его с вложеным списком и попробуйте снова.")
        return

    # Обработка данных (без изменений)
    close_obj = data["Close"] if "Close" in data.columns else data.get("Close")
    if isinstance(close_obj, pd.DataFrame):
        num_cols = close_obj.select_dtypes(include=[np.number])
        if num_cols.shape[1] == 0:
            logger.warning(f"В колонке Close нет числовых данных для {ticker}")
            await user.bot.send_message(chat_id=user.id, text="В колонке Close нет числовых данных. Попробуйте другой тикер или выберите из списка.")
            return
        close = num_cols.iloc[:, 0].copy()
    else:
        close = pd.Series(close_obj, copy=True)

    close = close.dropna()
    close.index = pd.to_datetime(close.index)
    close = pd.Series(np.asarray(close).squeeze(), index=close.index, dtype="float64")
    
    # Удаляем выбросы
    close = remove_outliers(close)
    
    logger.info(f"Обработанные данные для {ticker}: len={len(close)}, last_date={close.index[-1] if not close.empty else 'N/A'}")

    if len(close) < 80:
        logger.warning(f"Недостаточно данных для обучения {ticker}: len={len(close)} < 80")
        await user.bot.send_message(chat_id=user.id, text="Истории недостаточно для обучения. Нужен хотя бы квартал плотных данных.")
        return

    test_days = 30
    results: List[EvalResult] = []

    try:
        logger.info(f"Обучение RIDGE для {ticker}")
        results.append(ridge_fit_predict(close, test_days))
        logger.info(f"RIDGE обучен для {ticker}")
    except Exception as e:
        logger.exception(f"RIDGE error для {ticker}: {e}")

    try:
        logger.info(f"Обучение ARIMA для {ticker}")
        results.append(arima_fit_predict(close, test_days))
        logger.info(f"ARIMA обучен для {ticker}")
    except Exception as e:
        logger.exception(f"ARIMA error для {ticker}: {e}")

    if TF_AVAILABLE:
        try:
            logger.info(f"Обучение LSTM для {ticker}")
            lstm_res = lstm_fit_predict(close, test_days)
            if lstm_res is not None:
                results.append(lstm_res)
                logger.info(f"LSTM обучен для {ticker}")
            else:
                logger.warning(f"LSTM вернул None для {ticker}")
        except Exception as e:
            logger.exception(f"LSTM error для {ticker}: {e}")

    if not results:
        logger.error(f"Все модели упали на обучении для {ticker}")
        await user.bot.send_message(chat_id=user.id, text="Все модели упали на обучении. Попробуйте другой тикер.")
        return

    best = pick_best_model(results)

    # Прогноз на 30 дней — ансамбль для лучших предсказаний
    forecast = ensemble_forecast(results, close, 30)

    # Изменение относительно последней цены
    last_price = float(close.iloc[-1])
    last_forecast = float(forecast.iloc[-1])
    delta_pct = ((last_forecast - last_price) / last_price) * 100.0

    # Дополнительная статистика
    volatility = calculate_volatility(close)
    trend = calculate_trend(close)

    # Рекомендации: локальные экстремумы прогноза
    lows, highs = local_extrema(forecast, window=2)
    profit_est, trades = simulate_strategy(last_price, forecast, amount)

    # График
    img_bytes = plot_history_forecast(close[-120:], forecast, lows, highs, ticker, volatility, trend, delta_pct)  # история только за ~полгода для читабельности

    # Текстовый ответ
    lines = []
    lines.append(f"Лучший алгоритм: *{best.name}*")
    lines.append(f"RMSE: `{best.rmse:.4f}`, MAPE: `{best.mape:.2f}%`")
    lines.append(f"Текущая цена: *{last_price:.2f} USD*")
    lines.append(f"Волатильность (год): `{volatility:.2f}%`")
    lines.append(f"Тренд (год): *{trend}*")
    lines.append(f"Прогноз на 30 дней: изменение к последнему дню: *{delta_pct:+.2f}%*")
    if trades:
        lines.append(f"Найдено сделок по сигналам: *{len(trades)}*")
        preview = "\n".join([f"• {b} → {s}: {pb:.2f} → {ps:.2f}" for (b, s, pb, ps) in trades[:5]])
        lines.append(preview)
    else:
        lines.append("Сигналы для покупок/продаж по локальным экстремумам не выражены.")
    lines.append(f"Ориентировочная прибыль от стратегии на сумму {amount:.2f}: *{profit_est:.2f}* у.е.")
    lines.append(f"*⚠️ Результаты носят учебный характер и не являются инвестсоветом.*")
    text = "\n".join(lines)

    # Отправка результата
    await user.bot.send_photo(
        chat_id=user.id,
        photo=types.BufferedInputFile(img_bytes, filename=f"{ticker}_forecast.png"),
        caption=text,
        parse_mode=ParseMode.MARKDOWN,
    )

    # Лог
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_id": str(user.id),
        "ticker": ticker,
        "amount": amount,
        "best_model": best.name,
        "rmse": round(best.rmse, 6),
        "mape": round(best.mape, 4),
        "delta_pct": round(delta_pct, 4),
        "profit_est": round(profit_est, 4),
    }
    try:
        safe_log(row, path="logs.csv")
    except Exception as e:
        logger.warning("log write failed: %s", e)

async def handle_query(message: types.Message) -> None:
    await process_query(message.text, message.from_user)

def main() -> None:
    token = TOKEN
    if not token:
        raise RuntimeError("Не найден TOKEN в переменных окружения. Создайте .env или экспортируйте токен.")
    bot = Bot(token=token)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start_cmd(message: types.Message) -> None:
        user_name = message.from_user.first_name or "Пользователь"
        await message.answer(
            f"🙂 Привет, {user_name}! Пришлите пожалуйста тикер и сумму. Например: `GOOGL 20000`\nИли выберите из списка в меню:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=make_kb(),
        )

    @dp.message(F.text)
    async def any_text(message: types.Message) -> None:
        user_id = message.from_user.id
        if user_id in user_states and user_states[user_id]["state"] == "waiting_for_amount":
            ticker = user_states[user_id]["ticker"]
            try:
                amount = float(message.text.strip())
                if amount <= 0:
                    raise ValueError
                del user_states[user_id]
                await process_query(f"{ticker} {amount}", message.from_user)
            except ValueError:
                await message.answer("Введите корректную сумму (число > 0):")
        else:
            await handle_query(message)

    @dp.callback_query(F.data)
    async def process_callback(callback: types.CallbackQuery) -> None:
        await callback.answer()
        data = callback.data
        if data == "show_tickers":
            await callback.message.edit_text("Выберите тикер:", reply_markup=make_tickers_kb())
        elif data == "back":
            await callback.message.edit_text(
                "Пришлите пожалуйста тикер и сумму. Например: `GOOGL 20000`\nИли выберите из списка в меню:",
                reply_markup=make_kb(),
            )
        elif data.startswith("ticker_"):
            ticker = data.split("_", 1)[1]
            user_id = callback.from_user.id
            user_states[user_id] = {"state": "waiting_for_amount", "ticker": ticker}
            await callback.message.edit_text(f"Выберите сумму для {ticker} в USD (например, 10000):", reply_markup=None)
        else:
            await process_query(data, callback.from_user)

    dp.run_polling(bot)

if __name__ == "__main__":
    main()
