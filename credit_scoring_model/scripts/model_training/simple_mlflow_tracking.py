"""
Упрощенный модуль для отслеживания экспериментов без MLflow.

Этот модуль предоставляет заглушки для функций MLflow,
чтобы код мог работать без установленного MLflow.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


class SimpleTracker:
    """
    Простой трекер для замены MLflow.
    """

    def __init__(
        self,
        experiment_name: str = "credit-scoring",
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        """
        Инициализация простого трекера.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:///./mlruns"
        self.registry_uri = registry_uri
        self.experiment_id = "0"
        self.current_run = None
        self.run_data = {}

        print(f"Используется простой трекер для эксперимента: {experiment_name}")

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ):
        """Начинает новый run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = {
            "run_id": run_id,
            "run_name": run_name or f"run_{run_id}",
            "tags": tags or {},
            "start_time": datetime.now(),
            "metrics": {},
            "params": {},
            "artifacts": [],
        }
        print(f"Начат run: {self.current_run['run_name']}")
        return self

    def end_run(self):
        """Завершает текущий run."""
        if self.current_run:
            self.current_run["end_time"] = datetime.now()
            duration = (
                self.current_run["end_time"] - self.current_run["start_time"]
            ).total_seconds()
            print(
                f"Завершен run: {self.current_run['run_name']} (длительность: {duration:.2f}с)"
            )
            self.run_data[self.current_run["run_id"]] = self.current_run.copy()
            self.current_run = None

    def log_param(self, key: str, value: Any):
        """Логирует параметр."""
        if self.current_run:
            self.current_run["params"][key] = value
            print(f"  Параметр: {key} = {value}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Логирует метрику."""
        if self.current_run:
            if key not in self.current_run["metrics"]:
                self.current_run["metrics"][key] = []
            self.current_run["metrics"][key].append({"value": value, "step": step})
            print(f"  Метрика: {key} = {value}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Логирует несколько метрик."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_data_info(self, X_train, X_test, y_train, y_test):
        """Логирует информацию о данных."""
        if self.current_run:
            data_info = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X_train.shape[1]
                if hasattr(X_train, "shape")
                else len(X_train.columns),
                "train_positive_rate": y_train.mean()
                if hasattr(y_train, "mean")
                else sum(y_train) / len(y_train),
                "test_positive_rate": y_test.mean()
                if hasattr(y_test, "mean")
                else sum(y_test) / len(y_test),
            }
            for key, value in data_info.items():
                self.log_param(f"data_{key}", value)
            print(f"  Информация о данных: {data_info}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Логирует артефакт."""
        if self.current_run:
            artifact_info = {
                "local_path": local_path,
                "artifact_path": artifact_path or os.path.basename(local_path),
                "timestamp": datetime.now().isoformat(),
            }
            self.current_run["artifacts"].append(artifact_info)
            print(f"  Артефакт: {local_path}")

    def log_model(self, model, artifact_path: str = "model", **kwargs):
        """Логирует модель."""
        if self.current_run:
            # Сохраняем модель локально
            model_path = (
                f"models/artifacts/{self.current_run['run_id']}_{artifact_path}.pkl"
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            self.log_artifact(model_path, artifact_path)
            print(f"  Модель сохранена: {model_path}")

    def set_tag(self, key: str, value: str):
        """Устанавливает тег."""
        if self.current_run:
            self.current_run["tags"][key] = value
            print(f"  Тег: {key} = {value}")

    def set_tags(self, tags: Dict[str, str]):
        """Устанавливает несколько тегов."""
        for key, value in tags.items():
            self.set_tag(key, value)

    def save_run_data(self):
        """Сохраняет данные всех runs в JSON файл."""
        output_path = f"models/artifacts/{self.experiment_name}_runs.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Конвертируем datetime объекты в строки для JSON
        serializable_data = {}
        for run_id, run_info in self.run_data.items():
            serializable_data[run_id] = {}
            for key, value in run_info.items():
                if isinstance(value, datetime):
                    serializable_data[run_id][key] = value.isoformat()
                else:
                    serializable_data[run_id][key] = value

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        print(f"Данные runs сохранены: {output_path}")


def setup_mlflow_experiment(experiment_name: str = "credit-scoring") -> SimpleTracker:
    """
    Настраивает эксперимент (заглушка для MLflow).

    Args:
        experiment_name: Название эксперимента

    Returns:
        SimpleTracker: Простой трекер
    """
    return SimpleTracker(experiment_name=experiment_name)


def log_model_experiment(
    model,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    metrics: dict,
    model_params: dict,
) -> str:
    """
    Заглушка для логирования эксперимента с моделью.

    Args:
        model: Обученная модель
        model_name: Название модели
        X_train: Обучающие данные
        X_test: Тестовые данные
        y_train: Обучающие метки
        y_test: Тестовые метки
        metrics: Метрики модели
        model_params: Параметры модели

    Returns:
        str: ID запуска (заглушка)
    """
    print(f"Логирование эксперимента для модели {model_name}")
    print(f"Метрики: {metrics}")
    print(f"Параметры: {model_params}")
    return "mock-run-id"


# Создаем алиасы для совместимости с MLflow
MLflowTracker = SimpleTracker
