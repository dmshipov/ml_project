"""Endpoints для метрик и мониторинга."""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.metrics_service import MetricsService

logger = structlog.get_logger()
router = APIRouter()


def get_metrics_service() -> MetricsService:
    """Получить экземпляр сервиса метрик."""
    return MetricsService()


@router.get("/metrics")
async def get_metrics(service: MetricsService = Depends(get_metrics_service)):
    """Получить метрики Prometheus."""
    try:
        metrics = await service.get_prometheus_metrics()
        return metrics

    except Exception as e:
        logger.error("Не удалось получить метрики", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Не удалось получить метрики: {str(e)}"
        )


@router.get("/metrics/predictions")
async def get_prediction_metrics(
    hours: int = Query(default=24, ge=1, le=168, description="Часов назад для анализа"),
    service: MetricsService = Depends(get_metrics_service),
):
    """Получить метрики предсказаний за указанный период времени."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        metrics = await service.get_prediction_metrics(start_time, end_time)
        return metrics

    except Exception as e:
        logger.error(
            "Не удалось получить метрики предсказаний", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось получить метрики предсказаний: {str(e)}",
        )


@router.get("/metrics/model-performance")
async def get_model_performance_metrics(
    service: MetricsService = Depends(get_metrics_service),
):
    """Получить метрики производительности модели."""
    try:
        metrics = await service.get_model_performance_metrics()
        return metrics

    except Exception as e:
        logger.error(
            "Не удалось получить метрики производительности модели",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось получить метрики производительности модели: {str(e)}",
        )


@router.get("/metrics/system")
async def get_system_metrics(service: MetricsService = Depends(get_metrics_service)):
    """Получить системные метрики."""
    try:
        metrics = await service.get_system_metrics()
        return metrics

    except Exception as e:
        logger.error(
            "Не удалось получить системные метрики", error=str(e), exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Не удалось получить системные метрики: {str(e)}"
        )


@router.get("/metrics/alerts")
async def get_active_alerts(service: MetricsService = Depends(get_metrics_service)):
    """Получить активные алерты."""
    try:
        alerts = await service.get_active_alerts()
        return alerts

    except Exception as e:
        logger.error("Не удалось получить активные алерты", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Не удалось получить активные алерты: {str(e)}"
        )
