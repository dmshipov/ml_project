"""
Модуль для интеграции с MLflow для отслеживания экспериментов.

Этот модуль предоставляет функции для:
1. Настройки MLflow
2. Логирования экспериментов
3. Сохранения моделей в MLflow Model Registry
4. Отслеживания метрик и параметров
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Проверяем доступность MLflow
try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    print(
        "Предупреждение: MLflow не установлен. Отслеживание экспериментов будет отключено."
    )
    MLFLOW_AVAILABLE = False
import json


class MLflowTracker:
    """
    Класс для работы с MLflow в контексте кредитного скоринга.
    """

    def __init__(
        self,
        experiment_name: str = "credit-scoring",
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        """
        Инициализация MLflow трекера.

        Args:
            experiment_name: Название эксперимента
            tracking_uri: URI для отслеживания (по умолчанию локальный)
            registry_uri: URI для реестра моделей
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        self.registry_uri = registry_uri

        if not MLFLOW_AVAILABLE:
            print("MLflow недоступен. Отслеживание экспериментов отключено.")
            self.experiment_id = "0"
            return

        try:
            # Настройка MLflow
            mlflow.set_tracking_uri(self.tracking_uri)
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)

            # Создание или получение эксперимента
            try:
                # Пытаемся получить существующий эксперимент
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    # Если эксперимент не существует, создаем его
                    self.experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    self.experiment_id = experiment.experiment_id
            except Exception as e:
                # Если есть проблемы с MLflow, используем дефолтный эксперимент
                print(f"Предупреждение: Не удалось настроить эксперимент MLflow: {e}")
                self.experiment_id = "0"  # Дефолтный эксперимент

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Ошибка инициализации MLflow: {e}")
            self.experiment_id = "0"

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Начинает новый run в MLflow.

        Args:
            run_name: Название run
            tags: Теги для run

        Returns:
            ActiveRun объект
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_data_info(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Логирует информацию о данных.

        Args:
            X_train: Обучающие признаки
            X_test: Тестовые признаки
            y_train: Обучающая целевая переменная
            y_test: Тестовая целевая переменная
        """
        # Основная информация о данных
        mlflow.log_params(
            {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features_count": X_train.shape[1],
                "train_positive_rate": y_train.mean(),
                "test_positive_rate": y_test.mean(),
            }
        )

        # Информация о признаках
        feature_info = {
            "numeric_features": X_train.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "categorical_features": X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }

        mlflow.log_params(
            {
                "numeric_features_count": len(feature_info["numeric_features"]),
                "categorical_features_count": len(feature_info["categorical_features"]),
            }
        )

        # Сохраняем детальную информацию о признаках
        mlflow.log_text(json.dumps(feature_info, indent=2), "feature_info.json")

    def log_model_params(self, model_params: Dict[str, Any]) -> None:
        """
        Логирует параметры модели.

        Args:
            model_params: Словарь с параметрами модели
        """
        # Фильтруем параметры для логирования
        filtered_params = {}
        for key, value in model_params.items():
            if isinstance(value, (str, int, float, bool)):
                filtered_params[key] = value
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                filtered_params[key] = str(value)

        mlflow.log_params(filtered_params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Логирует метрики модели.

        Args:
            metrics: Словарь с метриками
        """
        mlflow.log_metrics(metrics)

    def log_model(
        self,
        model: Any,
        model_name: str,
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Логирует модель в MLflow.

        Args:
            model: Обученная модель
            model_name: Название модели
            signature: Сигнатура модели
            input_example: Пример входных данных
        """
        # Определяем тип модели и используем соответствующий метод
        if hasattr(model, "predict_proba"):
            # Sklearn модель
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example,
            )
        else:
            # Общая модель
            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=model,
                signature=signature,
                input_example=input_example,
            )

    def log_artifacts(self, artifacts_dir: str) -> None:
        """
        Логирует артефакты (файлы).

        Args:
            artifacts_dir: Путь к папке с артефактами
        """
        mlflow.log_artifacts(artifacts_dir)

    def log_plots(self, plots: Dict[str, str]) -> None:
        """
        Логирует графики.

        Args:
            plots: Словарь {название: путь_к_файлу}
        """
        for plot_name, plot_path in plots.items():
            if Path(plot_path).exists():
                mlflow.log_artifact(plot_path, f"plots/{plot_name}")

    def register_model(
        self,
        model_name: str,
        model_version: str = "latest",
        description: Optional[str] = None,
    ) -> str:
        """
        Регистрирует модель в Model Registry.

        Args:
            model_name: Название модели
            model_version: Версия модели
            description: Описание модели

        Returns:
            URI зарегистрированной модели
        """
        try:
            # Получаем URI модели из текущего run
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"

            # Регистрируем модель
            registered_model = mlflow.register_model(
                model_uri=model_uri, name=model_name
            )

            # Добавляем описание
            if description:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=registered_model.version,
                    description=description,
                )

            return registered_model.name

        except Exception as e:
            print(f"Ошибка при регистрации модели: {e}")
            return None

    def get_best_model(
        self, metric_name: str = "roc_auc", ascending: bool = False
    ) -> Optional[Any]:
        """
        Получает лучшую модель по метрике.

        Args:
            metric_name: Название метрики
            ascending: Сортировка по возрастанию

        Returns:
            Лучшая модель или None
        """
        try:
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            )

            if runs:
                best_run = runs[0]
                # Здесь нужно будет загрузить модель из best_run
                return best_run
            else:
                return None

        except Exception as e:
            print(f"Ошибка при получении лучшей модели: {e}")
            return None

    def compare_models(
        self,
        run_ids: list,
        metric_names: list = ["accuracy", "precision", "recall", "f1", "roc_auc"],
    ) -> pd.DataFrame:
        """
        Сравнивает модели по метрикам.

        Args:
            run_ids: Список ID runs для сравнения
            metric_names: Список метрик для сравнения

        Returns:
            DataFrame с результатами сравнения
        """
        try:
            client = mlflow.tracking.MlflowClient()
            comparison_data = []

            for run_id in run_ids:
                run = client.get_run(run_id)
                run_data = {
                    "run_id": run_id,
                    "run_name": run.data.tags.get("mlflow.runName", run_id),
                    "start_time": run.info.start_time,
                }

                # Добавляем метрики
                for metric_name in metric_names:
                    if metric_name in run.data.metrics:
                        run_data[metric_name] = run.data.metrics[metric_name]
                    else:
                        run_data[metric_name] = None

                comparison_data.append(run_data)

            return pd.DataFrame(comparison_data)

        except Exception as e:
            print(f"Ошибка при сравнении моделей: {e}")
            return pd.DataFrame()


def setup_mlflow_experiment(experiment_name: str = "credit-scoring") -> MLflowTracker:
    """
    Настройка MLflow эксперимента.

    Args:
        experiment_name: Название эксперимента

    Returns:
        Настроенный MLflowTracker
    """
    return MLflowTracker(experiment_name=experiment_name)


def log_model_experiment(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metrics: Dict[str, float],
    model_params: Dict[str, Any],
    artifacts_dir: Optional[str] = None,
) -> str:
    """
    Логирует полный эксперимент с моделью.

    Args:
        model: Обученная модель
        model_name: Название модели
        X_train: Обучающие признаки
        X_test: Тестовые признаки
        y_train: Обучающая целевая переменная
        y_test: Тестовая целевая переменная
        metrics: Метрики модели
        model_params: Параметры модели
        artifacts_dir: Папка с артефактами

    Returns:
        ID run'а
    """
    tracker = setup_mlflow_experiment()

    with tracker.start_run(run_name=model_name) as run:
        # Логируем информацию о данных
        tracker.log_data_info(X_train, X_test, y_train, y_test)

        # Логируем параметры модели
        tracker.log_model_params(model_params)

        # Логируем метрики
        tracker.log_metrics(metrics)

        # Логируем модель
        tracker.log_model(model, model_name)

        # Логируем артефакты
        if artifacts_dir and Path(artifacts_dir).exists():
            tracker.log_artifacts(artifacts_dir)

        return run.info.run_id


if __name__ == "__main__":
    # Пример использования
    print("MLflow трекер для кредитного скоринга")
    print("Используйте этот модуль в ваших скриптах обучения моделей")
