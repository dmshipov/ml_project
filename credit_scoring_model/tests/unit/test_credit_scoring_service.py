"""Unit tests for credit scoring service."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from app.schemas.credit_scoring import CreditScoringRequest
from app.services.credit_scoring_service import CreditScoringService


class TestCreditScoringService:
    """Test cases for CreditScoringService."""

    @pytest.fixture
    def mock_model(self):
        """Mock ML model for testing."""
        model = Mock()
        model.predict_proba.return_value = np.array(
            [[0.3, 0.7]]
        )  # 70% probability of rejection
        model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        return model

    @pytest.fixture
    def service_with_mock_model(self, mock_model, test_settings):
        """Service with mocked model."""
        with patch("joblib.load", return_value=mock_model), patch(
            "app.services.credit_scoring_service.get_settings",
            return_value=test_settings,
        ):
            service = CreditScoringService()
            service.model = mock_model
            return service

    def test_prepare_features(self, service_with_mock_model):
        """Test feature preparation."""
        request = CreditScoringRequest(
            annual_inc=50000.0,
            emp_length="5 years",
            home_ownership="MORTGAGE",
            loan_amnt=10000.0,
            term="36 months",
            purpose="debt_consolidation",
            fico_range_low=700,
            fico_range_high=750,
            dti=15.5,
            revol_util=25.0,
            inq_last_6mths=2,
            delinq_2yrs=0,
            pub_rec=0,
        )

        features_df = service_with_mock_model._prepare_features(request)

        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 1
        assert "annual_inc" in features_df.columns
        assert "loan_amnt" in features_df.columns
        assert features_df["annual_inc"].iloc[0] == 50000.0
        assert features_df["loan_amnt"].iloc[0] == 10000.0

    def test_handle_missing_values(self, service_with_mock_model):
        """Test missing value handling."""
        features = {
            "annual_inc": 50000.0,
            "revol_util": None,
            "inq_last_6mths": None,
            "emp_length": None,
        }

        processed = service_with_mock_model._handle_missing_values(features)

        assert processed["revol_util"] == 0.0
        assert processed["inq_last_6mths"] == 0
        assert processed["emp_length"] == "n/a"
        assert processed["annual_inc"] == 50000.0

    def test_convert_emp_length_to_numeric(self, service_with_mock_model):
        """Test employment length conversion."""
        assert service_with_mock_model._convert_emp_length_to_numeric("5 years") == 5
        assert service_with_mock_model._convert_emp_length_to_numeric("< 1 year") == 0
        assert service_with_mock_model._convert_emp_length_to_numeric("10+ years") == 10
        assert service_with_mock_model._convert_emp_length_to_numeric("n/a") == 0
        assert service_with_mock_model._convert_emp_length_to_numeric("invalid") == 0

    def test_get_confidence_level(self, service_with_mock_model):
        """Test confidence level calculation."""
        assert service_with_mock_model._get_confidence_level(0.9) == "high"
        assert service_with_mock_model._get_confidence_level(0.1) == "high"
        assert service_with_mock_model._get_confidence_level(0.6) == "medium"
        assert service_with_mock_model._get_confidence_level(0.4) == "medium"
        assert service_with_mock_model._get_confidence_level(0.5) == "low"

    def test_calculate_risk_score(self, service_with_mock_model):
        """Test risk score calculation."""
        assert service_with_mock_model._calculate_risk_score(0.7) == 70.0
        assert service_with_mock_model._calculate_risk_score(0.3) == 30.0
        assert service_with_mock_model._calculate_risk_score(0.0) == 0.0
        assert service_with_mock_model._calculate_risk_score(1.0) == 100.0

    def test_calculate_recommended_amount(self, service_with_mock_model):
        """Test recommended amount calculation."""
        request = CreditScoringRequest(
            annual_inc=50000.0,
            emp_length="5 years",
            home_ownership="MORTGAGE",
            loan_amnt=10000.0,
            term="36 months",
            purpose="debt_consolidation",
            fico_range_low=700,
            fico_range_high=750,
            dti=15.5,
        )

        # High risk - should reduce amount by 50%
        recommended = service_with_mock_model._calculate_recommended_amount(
            request, 0.8
        )
        assert recommended == 5000.0

        # Medium risk - should reduce amount by 20%
        recommended = service_with_mock_model._calculate_recommended_amount(
            request, 0.6
        )
        assert recommended == 8000.0

        # Low risk - should give full amount
        recommended = service_with_mock_model._calculate_recommended_amount(
            request, 0.3
        )
        assert recommended == 10000.0

    @pytest.mark.asyncio
    async def test_predict_credit_score(self, service_with_mock_model):
        """Test credit score prediction."""
        request = CreditScoringRequest(
            annual_inc=50000.0,
            emp_length="5 years",
            home_ownership="MORTGAGE",
            loan_amnt=10000.0,
            term="36 months",
            purpose="debt_consolidation",
            fico_range_low=700,
            fico_range_high=750,
            dti=15.5,
            revol_util=25.0,
            inq_last_6mths=2,
            delinq_2yrs=0,
            pub_rec=0,
        )

        prediction = await service_with_mock_model.predict_credit_score(request)

        assert (
            prediction.prediction == 1
        )  # Should be rejected (probability > threshold)
        assert prediction.probability == 0.7
        assert prediction.confidence == "medium"  # 0.7 falls in medium range (0.6-0.8)
        assert prediction.risk_score == 70.0
        assert prediction.model_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_model_info(self, service_with_mock_model):
        """Test model info retrieval."""
        info = await service_with_mock_model.get_model_info()

        assert info["model_version"] == "1.0.0"
        assert info["model_type"] == "Mock"
        assert info["threshold"] == 0.5
        assert "last_updated" in info

    @pytest.mark.asyncio
    async def test_get_prediction_stats(self, service_with_mock_model):
        """Test prediction stats retrieval."""
        stats = await service_with_mock_model.get_prediction_stats()

        assert "total_predictions" in stats
        assert "approval_rate" in stats
        assert "average_processing_time_ms" in stats
        assert "last_24h_predictions" in stats
