"""
Unit тесты для API (api/simple_main.py).
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.simple_main import app


class TestAPI:
    """Тесты для API endpoints."""

    @pytest.fixture
    def client(self):
        """Создает тестовый клиент FastAPI."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Тест главной страницы API."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert data["version"] == "1.0.0"

    def test_health_check(self, client):
        """Тест health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert "database_status" in data
        assert "model_status" in data
        assert "uptime_seconds" in data

    def test_predict_endpoint_valid_request(self, client):
        """Тест predict endpoint с валидным запросом."""
        request_data = {
            "annual_inc": 50000.0,
            "emp_length": "5 years",
            "home_ownership": "MORTGAGE",
            "loan_amnt": 10000.0,
            "term": "36 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 700,
            "fico_range_high": 750,
            "dti": 15.5,
            "revol_util": 25.0,
            "inq_last_6mths": 2,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert "processing_time_ms" in data
        assert "request_id" in data
        assert "timestamp" in data

        # Проверяем структуру prediction
        prediction = data["prediction"]
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "confidence" in prediction
        assert "risk_score" in prediction
        assert "recommended_amount" in prediction
        assert "model_version" in prediction
        assert "features_importance" in prediction

        # Проверяем типы данных
        assert isinstance(prediction["prediction"], int)
        assert isinstance(prediction["probability"], float)
        assert isinstance(prediction["risk_score"], float)
        assert isinstance(prediction["recommended_amount"], float)
        assert isinstance(prediction["features_importance"], dict)

    def test_predict_endpoint_high_risk(self, client):
        """Тест predict endpoint с высоким риском."""
        request_data = {
            "annual_inc": 20000.0,
            "emp_length": "1 year",
            "home_ownership": "RENT",
            "loan_amnt": 50000.0,
            "term": "60 months",
            "purpose": "other",
            "fico_range_low": 500,
            "fico_range_high": 550,
            "dti": 40.0,
            "revol_util": 80.0,
            "inq_last_6mths": 5,
            "delinq_2yrs": 2,
            "pub_rec": 1,
        }

        response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        prediction = data["prediction"]
        assert prediction["prediction"] == 1  # Rejected
        # Проверяем, что вероятность соответствует логике API
        assert 0.0 <= prediction["probability"] <= 1.0
        assert prediction["confidence"] in ["high", "medium", "low"]
        assert 0.0 <= prediction["risk_score"] <= 100.0

    def test_predict_endpoint_low_risk(self, client):
        """Тест predict endpoint с низким риском."""
        request_data = {
            "annual_inc": 80000.0,
            "emp_length": "10+ years",
            "home_ownership": "OWN",
            "loan_amnt": 5000.0,
            "term": "36 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 750,
            "fico_range_high": 800,
            "dti": 10.0,
            "revol_util": 15.0,
            "inq_last_6mths": 0,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        prediction = data["prediction"]
        assert prediction["prediction"] == 0  # Approved
        # Проверяем, что вероятность соответствует логике API
        assert 0.0 <= prediction["probability"] <= 1.0
        assert prediction["confidence"] in ["high", "medium", "low"]
        assert 0.0 <= prediction["risk_score"] <= 100.0

    def test_predict_endpoint_invalid_data(self, client):
        """Тест predict endpoint с невалидными данными."""
        # Тест с отсутствующими полями
        request_data = {
            "annual_inc": 50000.0,
            "emp_length": "5 years"
            # Отсутствуют обязательные поля
        }

        response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_wrong_types(self, client):
        """Тест predict endpoint с неправильными типами данных."""
        request_data = {
            "annual_inc": "fifty thousand",  # Должно быть float
            "emp_length": "5 years",
            "home_ownership": "MORTGAGE",
            "loan_amnt": 10000.0,
            "term": "36 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 700,
            "fico_range_high": 750,
            "dti": 15.5,
            "revol_util": 25.0,
            "inq_last_6mths": 2,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_edge_cases(self, client):
        """Тест predict endpoint с граничными случаями."""
        # Минимальные значения
        request_data = {
            "annual_inc": 0.0,
            "emp_length": "< 1 year",
            "home_ownership": "RENT",
            "loan_amnt": 1.0,
            "term": "12 months",
            "purpose": "other",
            "fico_range_low": 300,
            "fico_range_high": 350,
            "dti": 0.0,
            "revol_util": 0.0,
            "inq_last_6mths": 0,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 200

        # Максимальные значения
        request_data = {
            "annual_inc": 1000000.0,
            "emp_length": "10+ years",
            "home_ownership": "OWN",
            "loan_amnt": 100000.0,
            "term": "60 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 850,
            "fico_range_high": 900,
            "dti": 50.0,
            "revol_util": 100.0,
            "inq_last_6mths": 10,
            "delinq_2yrs": 5,
            "pub_rec": 3,
        }

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 200

    def test_api_docs_endpoint(self, client):
        """Тест endpoint документации API."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200

    def test_response_time(self, client):
        """Тест времени ответа API."""
        import time

        request_data = {
            "annual_inc": 50000.0,
            "emp_length": "5 years",
            "home_ownership": "MORTGAGE",
            "loan_amnt": 10000.0,
            "term": "36 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 700,
            "fico_range_high": 750,
            "dti": 15.5,
            "revol_util": 25.0,
            "inq_last_6mths": 2,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        start_time = time.time()
        response = client.post("/api/v1/predict", json=request_data)
        end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time

        # Проверяем, что ответ пришел быстро (менее 1 секунды)
        assert response_time < 1.0

        # Проверяем, что processing_time_ms в ответе разумный
        data = response.json()
        assert data["processing_time_ms"] < 1000  # Менее 1 секунды в миллисекундах


class TestAPIIntegration:
    """Интеграционные тесты для API."""

    @pytest.fixture
    def client(self):
        """Создает тестовый клиент FastAPI."""
        return TestClient(app)

    def test_multiple_requests(self, client):
        """Тест множественных запросов к API."""
        request_data = {
            "annual_inc": 50000.0,
            "emp_length": "5 years",
            "home_ownership": "MORTGAGE",
            "loan_amnt": 10000.0,
            "term": "36 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 700,
            "fico_range_high": 750,
            "dti": 15.5,
            "revol_util": 25.0,
            "inq_last_6mths": 2,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        # Отправляем несколько запросов
        for i in range(5):
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_concurrent_requests(self, client):
        """Тест конкурентных запросов к API."""
        import threading
        import time

        request_data = {
            "annual_inc": 50000.0,
            "emp_length": "5 years",
            "home_ownership": "MORTGAGE",
            "loan_amnt": 10000.0,
            "term": "36 months",
            "purpose": "debt_consolidation",
            "fico_range_low": 700,
            "fico_range_high": 750,
            "dti": 15.5,
            "revol_util": 25.0,
            "inq_last_6mths": 2,
            "delinq_2yrs": 0,
            "pub_rec": 0,
        }

        results = []

        def make_request():
            response = client.post("/api/v1/predict", json=request_data)
            results.append(response.status_code)

        # Создаем несколько потоков
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()

        # Проверяем, что все запросы успешны
        assert all(status == 200 for status in results)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__])
