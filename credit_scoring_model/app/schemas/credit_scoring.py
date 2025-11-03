"""Pydantic схемы для запросов и ответов кредитного скоринга."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class CreditScoringRequest(BaseModel):
    """Схема запроса для предсказания кредитного скоринга."""

    # Личная информация
    annual_inc: float = Field(..., description="Годовой доход", gt=0)
    emp_length: Optional[str] = Field(None, description="Стаж работы")
    home_ownership: str = Field(..., description="Статус владения жильем")

    # Информация о займе
    loan_amnt: float = Field(..., description="Запрашиваемая сумма займа", gt=0)
    term: str = Field(..., description="Срок займа")
    purpose: str = Field(..., description="Цель займа")

    # Кредитная информация
    fico_range_low: int = Field(
        ..., description="Нижний диапазон FICO скора", ge=300, le=850
    )
    fico_range_high: int = Field(
        ..., description="Верхний диапазон FICO скора", ge=300, le=850
    )
    dti: float = Field(..., description="Соотношение долга к доходу", ge=0, le=100)

    # Дополнительные признаки
    revol_util: Optional[float] = Field(
        None, description="Использование возобновляемого кредита", ge=0, le=100
    )
    inq_last_6mths: Optional[int] = Field(
        None, description="Запросы за последние 6 месяцев", ge=0
    )
    delinq_2yrs: Optional[int] = Field(
        None, description="Просрочки за последние 2 года", ge=0
    )
    pub_rec: Optional[int] = Field(None, description="Публичные записи", ge=0)

    @validator("fico_range_high")
    def fico_range_high_must_be_greater_than_low(cls, v, values):
        """Проверить, что верхний FICO больше нижнего."""
        if "fico_range_low" in values and v <= values["fico_range_low"]:
            raise ValueError("fico_range_high должен быть больше fico_range_low")
        return v

    @validator("emp_length")
    def validate_emp_length(cls, v):
        """Проверить формат стажа работы."""
        if v is not None:
            valid_values = [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years",
                "n/a",
            ]
            if v not in valid_values:
                raise ValueError(f"emp_length должен быть одним из: {valid_values}")
        return v

    @validator("home_ownership")
    def validate_home_ownership(cls, v):
        """Проверить статус владения жильем."""
        valid_values = ["RENT", "OWN", "MORTGAGE", "OTHER"]
        if v not in valid_values:
            raise ValueError(f"home_ownership должен быть одним из: {valid_values}")
        return v

    @validator("term")
    def validate_term(cls, v):
        """Проверить срок займа."""
        valid_values = ["36 months", "60 months"]
        if v not in valid_values:
            raise ValueError(f"term должен быть одним из: {valid_values}")
        return v


class ModelPrediction(BaseModel):
    """Детали предсказания модели."""

    prediction: int = Field(..., description="Предсказание (0=одобрено, 1=отклонено)")
    probability: float = Field(..., description="Вероятность предсказания", ge=0, le=1)
    confidence: str = Field(..., description="Уровень уверенности")
    recommended_amount: Optional[float] = Field(
        None, description="Рекомендуемая сумма займа"
    )
    risk_score: float = Field(..., description="Оценка риска", ge=0, le=100)
    model_version: str = Field(..., description="Используемая версия модели")
    features_importance: Optional[Dict[str, float]] = Field(
        None, description="Оценки важности признаков"
    )

    class Config:
        """Конфигурация Pydantic."""

        json_encoders = {
            # Пользовательские кодировщики при необходимости
        }


class CreditScoringResponse(BaseModel):
    """Схема ответа для предсказания кредитного скоринга."""

    success: bool = Field(..., description="Статус успешности запроса")
    prediction: ModelPrediction = Field(..., description="Детали предсказания модели")
    processing_time_ms: float = Field(
        ..., description="Время обработки в миллисекундах"
    )
    request_id: str = Field(..., description="Уникальный идентификатор запроса")
    timestamp: str = Field(..., description="Временная метка предсказания")

    class Config:
        """Конфигурация Pydantic."""

        json_encoders = {
            # Пользовательские кодировщики при необходимости
        }


class CreditScoringFeedback(BaseModel):
    """Схема обратной связи для улучшения модели."""

    request_id: str = Field(..., description="Исходный идентификатор запроса")
    actual_outcome: int = Field(
        ..., description="Фактический результат (0=одобрено, 1=отклонено)"
    )
    feedback_notes: Optional[str] = Field(
        None, description="Дополнительные заметки обратной связи"
    )
    user_id: Optional[str] = Field(
        None, description="Идентификатор пользователя, предоставившего обратную связь"
    )

    @validator("actual_outcome")
    def validate_actual_outcome(cls, v):
        """Проверить фактический результат."""
        if v not in [0, 1]:
            raise ValueError("actual_outcome должен быть 0 или 1")
        return v


class HealthCheckResponse(BaseModel):
    """Схема ответа проверки состояния."""

    status: str = Field(..., description="Статус сервиса")
    timestamp: str = Field(..., description="Временная метка проверки")
    version: str = Field(..., description="Версия приложения")
    database_status: str = Field(..., description="Статус подключения к базе данных")
    model_status: str = Field(..., description="Статус загрузки модели")
    uptime_seconds: float = Field(..., description="Время работы сервиса в секундах")
