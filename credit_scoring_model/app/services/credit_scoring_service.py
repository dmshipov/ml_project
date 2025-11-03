"""Сервис кредитного скоринга для ML предсказаний."""

from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import structlog

from app.schemas.credit_scoring import (
    CreditScoringFeedback,
    CreditScoringRequest,
    ModelPrediction,
)
from config.settings import get_settings

logger = structlog.get_logger()


class CreditScoringService:
    """Сервис для предсказаний кредитного скоринга."""

    def __init__(self):
        """Инициализация сервиса кредитного скоринга."""
        self.settings = get_settings()
        self.model = None
        self.model_version = self.settings.model_version
        self.model_threshold = self.settings.model_threshold
        self._load_model()

    def _load_model(self):
        """Загрузить обученную ML модель."""
        try:
            model_path = self.settings.model_path
            self.model = joblib.load(model_path)
            logger.info("Модель успешно загружена", model_path=model_path)
        except Exception as e:
            logger.error(
                "Не удалось загрузить модель",
                error=str(e),
                model_path=self.settings.model_path,
            )
            raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

    async def predict_credit_score(
        self, request: CreditScoringRequest
    ) -> ModelPrediction:
        """Предсказать кредитный скоринг для данного запроса."""
        try:
            # Конвертировать запрос в DataFrame
            features_df = self._prepare_features(request)

            # Выполнить предсказание
            prediction_proba = self.model.predict_proba(features_df)[0]
            prediction = int(prediction_proba[1] >= self.model_threshold)
            probability = float(prediction_proba[1])

            # Определить уровень уверенности
            confidence = self._get_confidence_level(probability)

            # Вычислить оценку риска (0-100)
            risk_score = self._calculate_risk_score(probability)

            # Получить рекомендуемую сумму, если одобрено
            recommended_amount = None
            if prediction == 0:  # Одобрено
                recommended_amount = self._calculate_recommended_amount(
                    request, probability
                )

            # Получить важность признаков, если доступно
            feature_importance = self._get_feature_importance(features_df)

            return ModelPrediction(
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                recommended_amount=recommended_amount,
                risk_score=risk_score,
                model_version=self.model_version,
                features_importance=feature_importance,
            )

        except Exception as e:
            logger.error("Предсказание не удалось", error=str(e), exc_info=True)
            raise RuntimeError(f"Предсказание не удалось: {str(e)}")

    def _prepare_features(self, request: CreditScoringRequest) -> pd.DataFrame:
        """Подготовить признаки для предсказания модели."""
        # Конвертировать запрос в словарь
        features = request.dict()

        # Обработать пропущенные значения
        features = self._handle_missing_values(features)

        # Инженерия признаков
        features = self._engineer_features(features)

        # Конвертировать в DataFrame
        df = pd.DataFrame([features])

        # Обеспечить правильный порядок столбцов (должен соответствовать обучающим данным)
        # Это нужно будет обновить в зависимости от ваших реальных признаков модели
        expected_columns = [
            "annual_inc",
            "loan_amnt",
            "fico_range_low",
            "fico_range_high",
            "dti",
            "revol_util",
            "inq_last_6mths",
            "delinq_2yrs",
            "pub_rec",
        ]

        # Добавить отсутствующие столбцы со значениями по умолчанию
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Переупорядочить столбцы
        df = df[expected_columns]

        return df

    def _handle_missing_values(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values in features."""
        # Fill missing values with defaults
        defaults = {
            "revol_util": 0.0,
            "inq_last_6mths": 0,
            "delinq_2yrs": 0,
            "pub_rec": 0,
            "emp_length": "n/a",
        }

        for key, default_value in defaults.items():
            if key in features and features[key] is None:
                features[key] = default_value

        return features

    def _engineer_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer additional features."""
        # Calculate FICO score average
        if "fico_range_low" in features and "fico_range_high" in features:
            features["fico_avg"] = (
                features["fico_range_low"] + features["fico_range_high"]
            ) / 2

        # Calculate loan to income ratio
        if (
            "loan_amnt" in features
            and "annual_inc" in features
            and features["annual_inc"] > 0
        ):
            features["loan_to_income_ratio"] = (
                features["loan_amnt"] / features["annual_inc"]
            )
        else:
            features["loan_to_income_ratio"] = 0

        # Convert employment length to numeric
        if "emp_length" in features:
            features["emp_length_numeric"] = self._convert_emp_length_to_numeric(
                features["emp_length"]
            )

        return features

    def _convert_emp_length_to_numeric(self, emp_length: str) -> int:
        """Convert employment length string to numeric value."""
        if emp_length == "n/a" or emp_length is None:
            return 0

        if "< 1 year" in emp_length:
            return 0
        elif "10+ years" in emp_length:
            return 10
        else:
            # Extract number from string like "5 years"
            try:
                return int(emp_length.split()[0])
            except (ValueError, IndexError):
                return 0

    def _get_confidence_level(self, probability: float) -> str:
        """Get confidence level based on probability."""
        if probability >= 0.8 or probability <= 0.2:
            return "high"
        elif probability >= 0.6 or probability <= 0.4:
            return "medium"
        else:
            return "low"

    def _calculate_risk_score(self, probability: float) -> float:
        """Calculate risk score (0-100)."""
        return float(probability * 100)

    def _calculate_recommended_amount(
        self, request: CreditScoringRequest, probability: float
    ) -> Optional[float]:
        """Calculate recommended loan amount based on probability and risk."""
        if probability > 0.7:  # High risk
            return request.loan_amnt * 0.5  # Reduce amount by 50%
        elif probability > 0.5:  # Medium risk
            return request.loan_amnt * 0.8  # Reduce amount by 20%
        else:  # Low risk
            return request.loan_amnt  # Full amount

    def _get_feature_importance(
        self, features_df: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        try:
            if hasattr(self.model, "feature_importances_"):
                importance_dict = dict(
                    zip(features_df.columns, self.model.feature_importances_)
                )
                return importance_dict
            else:
                return None
        except Exception:
            return None

    async def log_prediction(
        self,
        request_id: str,
        request_data: Dict[str, Any],
        prediction_result: ModelPrediction,
        processing_time_ms: float,
    ):
        """Log prediction for monitoring and analytics."""
        try:
            # Log to database or external service
            # This is where you would implement actual logging
            logger.info(
                "Prediction logged",
                request_id=request_id,
                prediction=prediction_result.prediction,
                probability=prediction_result.probability,
                processing_time_ms=processing_time_ms,
            )
        except Exception as e:
            logger.error(
                "Failed to log prediction", error=str(e), request_id=request_id
            )

    async def process_feedback(self, feedback: CreditScoringFeedback):
        """Process feedback for model improvement."""
        try:
            # Store feedback for model retraining
            # This is where you would implement actual feedback processing
            logger.info(
                "Feedback processed",
                request_id=feedback.request_id,
                actual_outcome=feedback.actual_outcome,
            )
        except Exception as e:
            logger.error(
                "Failed to process feedback",
                error=str(e),
                request_id=feedback.request_id,
            )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_version": self.model_version,
            "model_type": type(self.model).__name__,
            "threshold": self.model_threshold,
            "features_count": len(self.model.feature_importances_)
            if hasattr(self.model, "feature_importances_")
            else None,
            "last_updated": datetime.utcnow().isoformat(),
        }

    async def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        # This would typically query a database for actual stats
        return {
            "total_predictions": 0,  # Placeholder
            "approval_rate": 0.0,  # Placeholder
            "average_processing_time_ms": 0.0,  # Placeholder
            "last_24h_predictions": 0,  # Placeholder
        }
