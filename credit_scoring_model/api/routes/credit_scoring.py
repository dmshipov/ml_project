"""Endpoints для предсказания кредитного скоринга."""

import time
import uuid
from datetime import datetime
from typing import Any, Dict

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.schemas.credit_scoring import (
    CreditScoringFeedback,
    CreditScoringRequest,
    CreditScoringResponse,
    ModelPrediction,
)
from app.services.credit_scoring_service import CreditScoringService
from config.settings import get_settings

logger = structlog.get_logger()
router = APIRouter()


def get_credit_scoring_service() -> CreditScoringService:
    """Получить экземпляр сервиса кредитного скоринга."""
    return CreditScoringService()


@router.post("/predict", response_model=CreditScoringResponse)
async def predict_credit_score(
    request: CreditScoringRequest,
    background_tasks: BackgroundTasks,
    service: CreditScoringService = Depends(get_credit_scoring_service),
):
    """Предсказать кредитный скоринг и одобрение займа."""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        logger.info(
            "Запрос на предсказание кредитного скоринга",
            request_id=request_id,
            loan_amount=request.loan_amnt,
            annual_income=request.annual_inc,
        )

        # Выполнить предсказание
        prediction_result = await service.predict_credit_score(request)

        # Вычислить время обработки
        processing_time = (
            time.time() - start_time
        ) * 1000  # Конвертировать в миллисекунды

        # Создать ответ
        response = CreditScoringResponse(
            success=True,
            prediction=prediction_result,
            processing_time_ms=processing_time,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Логировать результат предсказания
        logger.info(
            "Предсказание кредитного скоринга завершено",
            request_id=request_id,
            prediction=prediction_result.prediction,
            probability=prediction_result.probability,
            processing_time_ms=processing_time,
        )

        # Добавить фоновую задачу для логирования/метрик
        background_tasks.add_task(
            service.log_prediction,
            request_id=request_id,
            request_data=request.dict(),
            prediction_result=prediction_result,
            processing_time_ms=processing_time,
        )

        return response

    except Exception as e:
        logger.error(
            "Предсказание кредитного скоринга не удалось",
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Предсказание не удалось: {str(e)}"
        )


@router.post("/feedback")
async def submit_feedback(
    feedback: CreditScoringFeedback,
    background_tasks: BackgroundTasks,
    service: CreditScoringService = Depends(get_credit_scoring_service),
):
    """Отправить обратную связь для улучшения модели."""
    try:
        logger.info(
            "Получена обратная связь",
            request_id=feedback.request_id,
            actual_outcome=feedback.actual_outcome,
        )

        # Обработать обратную связь в фоне
        background_tasks.add_task(service.process_feedback, feedback=feedback)

        return {
            "success": True,
            "message": "Обратная связь успешно отправлена",
            "request_id": feedback.request_id,
        }

    except Exception as e:
        logger.error(
            "Отправка обратной связи не удалась",
            request_id=feedback.request_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Отправка обратной связи не удалась: {str(e)}"
        )


@router.get("/model/info")
async def get_model_info(
    service: CreditScoringService = Depends(get_credit_scoring_service),
):
    """Получить информацию о модели и статистику."""
    try:
        model_info = await service.get_model_info()
        return model_info

    except Exception as e:
        logger.error(
            "Не удалось получить информацию о модели", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Не удалось получить информацию о модели: {str(e)}"
        )


@router.get("/predictions/stats")
async def get_prediction_stats(
    service: CreditScoringService = Depends(get_credit_scoring_service),
):
    """Получить статистику предсказаний."""
    try:
        stats = await service.get_prediction_stats()
        return stats

    except Exception as e:
        logger.error(
            "Не удалось получить статистику предсказаний", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось получить статистику предсказаний: {str(e)}",
        )
