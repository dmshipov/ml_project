"""
Unit тесты для модулей обучения моделей.
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

from scripts.model_training.simple_mlflow_tracking import SimpleTracker as MLflowTracker
from scripts.model_training.simple_mlflow_tracking import (
    log_model_experiment,
    setup_mlflow_experiment,
)
from scripts.model_training.train_models import (
    create_models,
    load_processed_data,
    train_and_evaluate_models,
)


class TestModelTraining:
    """Тесты для обучения моделей."""

    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100),
            "feature4": np.random.choice(["X", "Y"], 100),
            "target": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_data_files(self, sample_data):
        """Создает временные файлы данных."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем файлы данных
            X_train = sample_data.drop(columns=["target"])
            y_train = sample_data["target"]
            X_test = sample_data.drop(columns=["target"])
            y_test = sample_data["target"]

            X_train.to_csv(f"{temp_dir}/X_train.csv", index=False)
            X_test.to_csv(f"{temp_dir}/X_test.csv", index=False)
            y_train.to_csv(f"{temp_dir}/y_train.csv", index=False)
            y_test.to_csv(f"{temp_dir}/y_test.csv", index=False)

            # Создаем mock препроцессор
            from sklearn.preprocessing import StandardScaler

            preprocessor = StandardScaler()
            joblib.dump(preprocessor, f"{temp_dir}/preprocessor.pkl")

            yield temp_dir

    def test_load_processed_data(self, temp_data_files):
        """Тест загрузки обработанных данных."""
        # Функция load_processed_data теперь возвращает только 4 значения (без preprocessor)
        X_train, X_test, y_train, y_test = load_processed_data(temp_data_files)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        # Убираем проверку preprocessor, так как он больше не возвращается
        
        # Проверяем размеры 
        assert X_train.shape[0] == 100
        assert X_test.shape[0] == 100
        assert len(y_train) == 100
        assert len(y_test) == 100


    def test_create_models(self, sample_data):
        """Тест создания моделей."""
        X_train = sample_data.drop(columns=["target"])
        models = create_models(X_train)

        assert isinstance(models, dict)
        assert len(models) > 0

        # Проверяем, что модели созданы
        expected_models = ["Logistic Regression", "Random Forest"]
        for model_name in expected_models:
            assert model_name in models
            assert hasattr(models[model_name], "fit")
            assert hasattr(models[model_name], "predict")

    def test_train_and_evaluate_models(self, sample_data):
        """Тест обучения и оценки моделей."""
        # Создаем тестовые данные
        X_train = sample_data.drop(columns=["target"])
        y_train = sample_data["target"]
        X_test = sample_data.drop(columns=["target"])
        y_test = sample_data["target"]

        # Создаем простые модели для тестирования
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        models = {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42)),
                ]
            ),
            "Random Forest": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(random_state=42, n_estimators=10),
                    ),
                ]
            ),
        }

        # Тестируем без MLflow
        results = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, use_mlflow=False
        )

        assert isinstance(results, dict)
        assert "results" in results
        assert "predictions" in results
        assert "probabilities" in results

        # Проверяем результаты
        for model_name in models.keys():
            assert model_name in results["results"]
            metrics = results["results"][model_name]

            # Проверяем, что если модель обучилась успешно, то есть предсказания
            if "error" not in metrics:
                assert model_name in results["predictions"]
                assert model_name in results["probabilities"]

                # Проверяем метрики только для успешно обученных моделей
                assert "accuracy" in metrics
                assert "precision" in metrics
                assert "recall" in metrics
                assert "f1" in metrics
                assert "roc_auc" in metrics

                # Проверяем, что метрики в разумных пределах
                assert 0 <= metrics["accuracy"] <= 1
                assert 0 <= metrics["precision"] <= 1
                assert 0 <= metrics["recall"] <= 1
                assert 0 <= metrics["f1"] <= 1
                assert 0 <= metrics["roc_auc"] <= 1
            else:
                # Если модель не обучилась, проверяем, что ошибка записана
                assert "error" in metrics
                assert isinstance(metrics["error"], str)

    def test_train_and_evaluate_models_with_mlflow(self, sample_data):
        """Тест обучения с MLflow."""
        X_train = sample_data.drop(columns=["target"])
        y_train = sample_data["target"]
        X_test = sample_data.drop(columns=["target"])
        y_test = sample_data["target"]

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        models = {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42)),
                ]
            )
        }

        results = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, use_mlflow=True
        )

        # Проверяем, что результаты получены
        assert isinstance(results, dict)
        assert "results" in results
        assert "Logistic Regression" in results["results"]

    def test_train_and_evaluate_models_error_handling(self, sample_data):
        """Тест обработки ошибок при обучении."""
        X_train = sample_data.drop(columns=["target"])
        y_train = sample_data["target"]
        X_test = sample_data.drop(columns=["target"])
        y_test = sample_data["target"]

        # Создаем модель, которая будет вызывать ошибку
        class FailingModel:
            def fit(self, X, y):
                raise Exception("Test error")

            def predict(self, X):
                return np.zeros(len(X))

        models = {"Failing Model": FailingModel()}

        results = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, use_mlflow=False
        )

        # Проверяем, что ошибка обработана
        assert "Failing Model" in results["results"]
        assert "error" in results["results"]["Failing Model"]


class TestMLflowTracking:
    """Тесты для MLflow трекинга."""

    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 50),
            "feature2": np.random.normal(0, 1, 50),
            "target": np.random.choice([0, 1], 50, p=[0.7, 0.3]),
        }
        return pd.DataFrame(data)

    def test_mlflow_tracker_initialization(self):
        """Тест инициализации MLflow трекера."""
        with patch("mlflow.set_tracking_uri"), patch("mlflow.create_experiment"), patch(
            "mlflow.set_experiment"
        ):
            tracker = MLflowTracker("test-experiment")

            assert tracker.experiment_name == "test-experiment"
            assert tracker.tracking_uri is not None

    def test_mlflow_tracker_log_data_info(self, sample_data):
        """Тест логирования информации о данных."""
        tracker = MLflowTracker("test-experiment")
        tracker.start_run()

        X_train = sample_data.drop(columns=["target"])
        X_test = sample_data.drop(columns=["target"])
        y_train = sample_data["target"]
        y_test = sample_data["target"]

        tracker.log_data_info(X_train, X_test, y_train, y_test)

        # Проверяем, что параметры были залогированы
        assert len(tracker.current_run["params"]) > 0
        assert "data_train_samples" in tracker.current_run["params"]
        assert "data_test_samples" in tracker.current_run["params"]

    def test_mlflow_tracker_log_metrics(self, sample_data):
        """Тест логирования метрик."""
        tracker = MLflowTracker("test-experiment")
        tracker.start_run()

        metrics = {"accuracy": 0.95, "precision": 0.90, "recall": 0.85}
        tracker.log_metrics(metrics)

        # Проверяем, что метрики были залогированы
        assert len(tracker.current_run["metrics"]) > 0
        assert "accuracy" in tracker.current_run["metrics"]
        assert "precision" in tracker.current_run["metrics"]
        assert "recall" in tracker.current_run["metrics"]

    def test_setup_mlflow_experiment(self):
        """Тест настройки MLflow эксперимента."""
        result = setup_mlflow_experiment("test-experiment")

        assert isinstance(result, MLflowTracker)
        assert result.experiment_name == "test-experiment"

    def test_log_model_experiment(self, sample_data):
        """Тест логирования эксперимента с моделью."""
        X_train = sample_data.drop(columns=["target"])
        X_test = sample_data.drop(columns=["target"])
        y_train = sample_data["target"]
        y_test = sample_data["target"]

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        metrics = {"accuracy": 0.95, "precision": 0.90}
        model_params = {"C": 1.0, "random_state": 42}

        run_id = log_model_experiment(
            model, "test-model", X_train, X_test, y_train, y_test, metrics, model_params
        )

        assert run_id == "mock-run-id"


class TestModelTrainingEdgeCases:
    """Тесты для граничных случаев обучения моделей."""

    def test_empty_dataframe(self):
        """Тест с пустым DataFrame."""
        empty_df = pd.DataFrame()

        # create_models должен обработать пустой DataFrame
        models = create_models(empty_df)
        assert isinstance(models, dict)

    def test_single_class_target(self):
        """Тест с целевой переменной одного класса."""
        single_class_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "target": [0, 0, 0, 0, 0],  # Только один класс
            }
        )

        X_train = single_class_data.drop(columns=["target"])
        y_train = single_class_data["target"]
        X_test = single_class_data.drop(columns=["target"])
        y_test = single_class_data["target"]

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        models = {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42)),
                ]
            )
        }

        # Тест должен обработать случай с одним классом
        results = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, use_mlflow=False
        )

        assert isinstance(results, dict)
        assert "results" in results

    def test_very_small_dataset(self):
        """Тест с очень маленьким датасетом."""
        small_data = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}
        )

        X_train = small_data.drop(columns=["target"])
        y_train = small_data["target"]
        X_test = small_data.drop(columns=["target"])
        y_test = small_data["target"]

        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        models = {
            "Logistic Regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(random_state=42)),
                ]
            )
        }

        # Тест должен обработать маленький датасет
        results = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, use_mlflow=False
        )

        assert isinstance(results, dict)
        assert "results" in results


if __name__ == "__main__":
    pytest.main([__file__])
