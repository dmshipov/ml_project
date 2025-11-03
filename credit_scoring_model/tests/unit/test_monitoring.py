"""
Unit тесты для модулей мониторинга.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.monitoring.data_quality_monitor import DataQualityMonitor
from scripts.monitoring.model_monitoring import ModelMonitor


class TestModelMonitoring:
    """Тесты для мониторинга моделей."""

    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),  # Изменено на числовое
            "feature4": np.random.normal(0, 1, 100),  # Изменено на числовое
            "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_model(self):
        """Создает mock модель."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=42)),
            ]
        )
        return model

    @pytest.fixture
    def temp_model_file(self, mock_model, sample_processed_data):
        """Создает временный файл модели."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            # Обучаем модель на тестовых данных
            X = sample_processed_data.drop(columns=["target"])
            y = sample_processed_data["target"]
            mock_model.fit(X, y)

            joblib.dump(mock_model, f.name)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_reference_data_file(self, sample_processed_data):
        """Создает временный файл референсных данных."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sample_processed_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_model_monitor_initialization(
        self, temp_model_file, temp_reference_data_file
    ):
        """Тест инициализации монитора модели."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)

        assert monitor.model_path == temp_model_file
        assert monitor.reference_data_path == temp_reference_data_file
        assert monitor.model is not None
        assert monitor.reference_data is not None

    def test_model_monitor_initialization_invalid_files(self):
        """Тест инициализации с неверными файлами."""
        with pytest.raises(Exception):
            ModelMonitor("nonexistent_model.pkl", "nonexistent_data.csv")

    def test_detect_data_drift_no_drift(
        self, temp_model_file, temp_reference_data_file, sample_processed_data
    ):
        """Тест детекции дрифта без дрифта."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)

        # Используем те же данные (без дрифта)
        feature_columns = ["feature1", "feature2", "feature3", "feature4"]
        result = monitor.detect_data_drift(sample_processed_data, feature_columns)

        assert isinstance(result, dict)
        assert "overall_drift_detected" in result
        assert "feature_drifts" in result
        assert "drift_score" in result
        assert "timestamp" in result

        # Дрифт не должен быть обнаружен
        assert result["drift_score"] < 0.5

    def test_generate_monitoring_report(
        self, temp_model_file, temp_reference_data_file, sample_processed_data
    ):
        """Тест генерации отчета мониторинга."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)

        feature_columns = ["feature1", "feature2", "feature3", "feature4"]
        X_test = sample_processed_data[feature_columns]
        y_test = sample_processed_data["target"]
        result = monitor.generate_monitoring_report(
            sample_processed_data, X_test, y_test, feature_columns
        )

        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "overall_status" in result
        assert "alerts" in result
        assert "model_path" in result


class TestDataQualityMonitoring:
    """Тесты для мониторинга качества данных."""

    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
            "feature4": np.random.choice(["X", "Y"], 100),
            "high_missing": [np.nan] * 70 + [1] * 30,  # 70% пропусков
            "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        }
        return pd.DataFrame(data)

    def test_data_quality_monitor_initialization(self):
        """Тест инициализации монитора качества данных."""
        monitor = DataQualityMonitor()
        assert monitor is not None

    def test_check_missing_values(self, sample_data):
        """Тест проверки пропущенных значений."""
        monitor = DataQualityMonitor()
        result = monitor.check_missing_values(sample_data)

        assert isinstance(result, dict)
        assert "missing_percentage" in result
        assert "columns_with_missing" in result
        assert "critical_columns" in result

        # Проверяем, что найдены колонки с пропусками
        assert len(result["columns_with_missing"]) > 0
        assert "high_missing" in [col["column"] for col in result["critical_columns"]]

    def test_check_data_distribution(self, sample_data):
        """Тест проверки распределения данных."""
        monitor = DataQualityMonitor()
        result = monitor.check_data_distribution(sample_data)

        assert isinstance(result, dict)
        assert "columns_distribution" in result
        assert "skewed_columns" in result
        assert "high_kurtosis_columns" in result

    def test_check_data_integrity(self, sample_data):
        """Тест проверки целостности данных."""
        monitor = DataQualityMonitor()
        result = monitor.check_data_integrity(sample_data)

        assert isinstance(result, dict)
        assert "duplicate_percentage" in result
        assert "constant_columns" in result
        assert "duplicate_columns" in result

    def test_check_categorical_data(self, sample_data):
        """Тест проверки категориальных данных."""
        monitor = DataQualityMonitor()
        result = monitor.check_categorical_data(sample_data)

        assert isinstance(result, dict)
        assert "categorical_columns" in result
        assert "high_cardinality_columns" in result
        assert "low_cardinality_columns" in result

    def test_generate_quality_report(self, sample_data):
        """Тест генерации отчета о качестве данных."""
        monitor = DataQualityMonitor()
        result = monitor.generate_quality_report(sample_data)

        assert isinstance(result, dict)
        assert "overall_quality_score" in result
        assert "missing_values" in result
        assert "data_integrity" in result
        assert "categorical_data" in result
        assert "recommendations" in result


class TestMonitoringEdgeCases:
    """Тесты для граничных случаев мониторинга."""

    def test_single_column_dataframe_monitoring(self):
        """Тест мониторинга с DataFrame с одной колонкой."""
        single_col_df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})

        monitor = DataQualityMonitor()
        result = monitor.generate_quality_report(single_col_df)

        assert isinstance(result, dict)
        assert "overall_quality_score" in result
        assert result["overall_quality_score"] >= 0
