"""
Скрипт для мониторинга качества моделей кредитного скоринга.

Этот скрипт выполняет:
1. Мониторинг дрифта данных (включая категориальные признаки)
2. Мониторинг производительности модели (с сравнением с baseline)
3. Мониторинг смещения предсказаний
4. Отправку алертов при деградации
"""

import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import io
import joblib
import sys
# Импорты для мониторинга
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# Импорты для статистических тестов
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Настройка логирования
# Создаем папку logs относительно корня проекта
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "model_monitoring.log", encoding='utf-8'),
        logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')),  # UTF-8 для консоли
    ],
    force=True  # Разрешить переопределение
)

# Переопределение logger для этого модуля
logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Класс для мониторинга качества моделей.
    """

    def __init__(
        self,
        model_path: str,
        reference_data_path: str,
        monitoring_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализация монитора модели.

        Args:
            model_path: Путь к обученной модели
            reference_data_path: Путь к референсным данным
            monitoring_config: Конфигурация мониторинга
        """
        self.model_path = model_path
        self.reference_data_path = reference_data_path
        self.config = monitoring_config or self._default_config()

        # Загружаем модель и референсные данные
        self.model = self._load_model()
        self.reference_data = self._load_reference_data()

        # Настройка MLflow
        self.mlflow_client = MlflowClient()

        logger.info(f"Монитор модели инициализирован: {model_path}")

    def _default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию мониторинга по умолчанию."""
        return {
            "drift_threshold": 0.1,  # Порог для детекции дрифта
            "performance_threshold": 0.05,  # Процент деградации производительности (например, 5%)
            "bias_threshold": 0.1,  # Порог для детекции смещения
            "min_samples": 100,  # Минимальное количество образцов для анализа
            "baseline_metrics": {
                "f1_score": 0.8,  # Базовые значения для сравнения (в проде загрузите из файла)
                "accuracy": 0.85,
                "precision": 0.8,
                "recall": 0.8,
            },  # Добавлен baseline
            "alert_email": None,  # Email для алертов
            "slack_webhook": None,  # Slack webhook для алертов
            "monitoring_window": 7,  # Окно мониторинга в днях
        }

    def _load_model(self):
        """Загружает обученную модель."""
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Модель загружена: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def _load_reference_data(self) -> pd.DataFrame:
        """Загружает референсные данные."""
        try:
            data = pd.read_csv(self.reference_data_path)
            logger.info(f"Референсные данные загружены: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Ошибка загрузки референсных данных: {e}")
            raise

    def detect_data_drift(
        self, current_data: pd.DataFrame, feature_columns: list
    ) -> Dict[str, Any]:
        """
        Детектирует дрифт данных между референсными и текущими данными.

        Args:
            current_data: Текущие данные
            feature_columns: Список признаков для анализа

        Returns:
            Словарь с результатами анализа дрифта
        """
        logger.info("Анализ дрифта данных...")

        drift_results = {
            "overall_drift_detected": False,
            "feature_drifts": {},
            "drift_score": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            drift_scores = []

            for feature in feature_columns:
                if feature in self.reference_data.columns:
                    ref_values = self.reference_data[feature].dropna()
                    curr_values = current_data[feature].dropna()

                    if ref_values.dtype in ['int64', 'float64']:
                        # KS тест для числовых
                        statistic, p_value = ks_2samp(ref_values, curr_values)
                    elif ref_values.dtype == 'object':
                        # Chi2 тест для категориальных
                        # Создаем contingency table для распределений
                        ref_counts = ref_values.value_counts()
                        curr_counts = curr_values.value_counts()
                        # Объединяем категории для таблицы
                        all_categories = set(ref_counts.index) | set(curr_counts.index)
                        ref_table = pd.Series(data=ref_counts, index=all_categories).fillna(0)
                        curr_table = pd.Series(data=curr_counts, index=all_categories).fillna(0)
                        cont_table = pd.DataFrame({'ref': ref_table, 'curr': curr_table}).values
                        chi2, p_value, _, _ = chi2_contingency(cont_table)
                        statistic = chi2
                    else:
                        continue  # Пропустить неподдерживаемые типы

                    drift_detected = p_value < self.config["drift_threshold"]
                    drift_scores.append(statistic)

                    drift_results["feature_drifts"][feature] = {
                        "drift_detected": drift_detected,
                        "statistic": statistic,  # KS или Chi2
                        "p_value": p_value,
                        "reference_mean_or_counts": ref_values.mean() if ref_values.dtype in ['int64', 'float64'] else ref_values.value_counts().to_dict(),
                        "current_mean_or_counts": curr_values.mean() if curr_values.dtype in ['int64', 'float64'] else curr_values.value_counts().to_dict(),
                    }

            # Общий скор дрифта
            if drift_scores:
                drift_results["drift_score"] = np.mean(drift_scores)
                drift_results["overall_drift_detected"] = (
                    drift_results["drift_score"] > self.config["drift_threshold"]
                )

            logger.info(
                f"Дрифт данных: {drift_results['overall_drift_detected']}, "
                f"скор: {drift_results['drift_score']:.4f}"
            )

        except Exception as e:
            logger.error(f"Ошибка при анализе дрифта: {e}")
            drift_results["error"] = str(e)

        return drift_results

    def monitor_model_performance(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Мониторит производительность модели с сравнением с baseline.

        Args:
            X_test: Тестовые признаки
            y_test: Тестовая целевая переменная

        Returns:
            Словарь с результатами мониторинга производительности
        """
        logger.info("Мониторинг производительности модели...")

        performance_results = {
            "current_metrics": {},
            "performance_degraded": False,
            "degradation_details": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Проверяем минимальное количество образцов
            if len(X_test) < self.config["min_samples"]:
                logger.warning(f"Недостаточно образцов: {len(X_test)} < {self.config['min_samples']}")
                return performance_results

            # Получаем предсказания
            y_pred = self.model.predict(X_test)
            y_proba = (
                self.model.predict_proba(X_test)[:, 1]
                if hasattr(self.model, "predict_proba")
                else None
            )

            # Вычисляем метрики
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            }

            if y_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

            performance_results["current_metrics"] = metrics

            # Проверяем деградацию с учетом baseline
            baseline = self.config.get("baseline_metrics", {})
            degraded_keys = []
            for key in ["f1_score", "accuracy", "precision", "recall"]:
                if key in baseline and key in metrics:
                    drop = (baseline[key] - metrics[key]) / baseline[key] if baseline[key] > 0 else 0
                    performance_results["degradation_details"][key] = drop
                    if drop > self.config["performance_threshold"]:
                        performance_results["performance_degraded"] = True
                        degraded_keys.append(key)

            if performance_results["performance_degraded"]:
                logger.warning(
                    f"Деградация производительности: превышение порога для {', '.join(degraded_keys)}"
                )
            else:
                logger.info("Производительность стабильна")

            logger.info(f"Текущие метрики: {metrics}")

        except Exception as e:
            logger.error(f"Ошибка при мониторинге производительности: {e}")
            performance_results["error"] = str(e)

        return performance_results

    def detect_prediction_bias(
        self, X_test: pd.DataFrame, y_test: pd.Series, sensitive_attributes: list
    ) -> Dict[str, Any]:
        """
        Детектирует смещение в предсказаниях модели.

        Args:
            X_test: Тестовые признаки
            y_test: Тестовая целевая переменная
            sensitive_attributes: Список чувствительных атрибутов

        Returns:
            Словарь с результатами анализа смещения
        """
        logger.info("Анализ смещения предсказаний...")

        bias_results = {
            "bias_detected": False,
            "attribute_bias": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            if len(X_test) < self.config["min_samples"]:
                logger.warning("Недостаточно образцов для анализа bias")
                return bias_results

            y_pred = self.model.predict(X_test)

            for attr in sensitive_attributes:
                if attr in X_test.columns:
                    # Группируем по значениям атрибута
                    groups = X_test[attr].unique()

                    if len(groups) > 1:
                        group_metrics = {}

                        for group in groups:
                            mask = X_test[attr] == group
                            if mask.sum() > 0:
                                group_y_true = y_test[mask]
                                group_y_pred = y_pred[mask]

                                group_metrics[group] = {
                                    "accuracy": accuracy_score(
                                        group_y_true, group_y_pred
                                    ),
                                    "precision": precision_score(
                                        group_y_true, group_y_pred, zero_division=0
                                    ),
                                    "recall": recall_score(
                                        group_y_true, group_y_pred, zero_division=0
                                    ),
                                    "f1_score": f1_score(
                                        group_y_true, group_y_pred, zero_division=0
                                    ),
                                    "sample_size": mask.sum(),
                                }

                        # Проверяем значительные различия между группами
                        if len(group_metrics) > 1:
                            accuracies = [
                                metrics["accuracy"]
                                for metrics in group_metrics.values()
                            ]
                            max_diff = max(accuracies) - min(accuracies)

                            bias_results["attribute_bias"][attr] = {
                                "group_metrics": group_metrics,
                                "max_accuracy_difference": max_diff,
                                "bias_detected": max_diff
                                > self.config["bias_threshold"],
                            }

                            if max_diff > self.config["bias_threshold"]:
                                bias_results["bias_detected"] = True
                                logger.warning(
                                    f"Смещение обнаружено для атрибута {attr}: "
                                    f"разница в точности {max_diff:.4f}"
                                )

            logger.info(f"Смещение предсказаний: {bias_results['bias_detected']}")

        except Exception as e:
            logger.error(f"Ошибка при анализе смещения: {e}")
            bias_results["error"] = str(e)

        return bias_results

    def generate_monitoring_report(
        self,
        current_data: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_columns: list,
        sensitive_attributes: list = None,
    ) -> Dict[str, Any]:
        """
        Генерирует полный отчет мониторинга.

        Args:
            current_data: Текущие данные
            X_test: Тестовые признаки
            y_test: Тестовая целевая переменная
            feature_columns: Список признаков
            sensitive_attributes: Список чувствительных атрибутов

        Returns:
            Полный отчет мониторинга
        """
        logger.info("Генерация отчета мониторинга...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "reference_data_path": self.reference_data_path,
            "alerts": [],
        }

        try:
            # Анализ дрифта данных
            drift_results = self.detect_data_drift(current_data, feature_columns)
            report["data_drift"] = drift_results

            if drift_results.get("overall_drift_detected", False):
                report["alerts"].append(
                    {
                        "type": "data_drift",
                        "severity": "high",
                        "message": "Обнаружен дрифт данных",
                        "details": f"Общий скор дрифта: {drift_results['drift_score']:.4f}",
                    }
                )

            # Мониторинг производительности
            performance_results = self.monitor_model_performance(X_test, y_test)
            report["performance"] = performance_results

            if performance_results.get("performance_degraded", False):
                report["alerts"].append(
                    {
                        "type": "performance_degradation",
                        "severity": "high",
                        "message": "Обнаружена деградация производительности",
                        "details": f"Перевышение порога для: {', '.join(performance_results.get('degradation_details', {}).keys())}",
                    }
                )

            # Анализ смещения
            if sensitive_attributes:
                bias_results = self.detect_prediction_bias(
                    X_test, y_test, sensitive_attributes
                )
                report["bias"] = bias_results

                if bias_results.get("bias_detected", False):
                    report["alerts"].append(
                        {
                            "type": "prediction_bias",
                            "severity": "medium",
                            "message": "Обнаружено смещение в предсказаниях",
                            "details": "Проверьте чувствительные атрибуты",
                        }
                    )

            # Общая оценка
            report["overall_status"] = (
                "healthy" if not report["alerts"] else "issues_detected"
            )

            logger.info(
                f"Отчет мониторинга сгенерирован: {len(report['alerts'])} алертов"
            )

        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {e}")
            report["error"] = str(e)
            report["overall_status"] = "error"

        return report

    def save_monitoring_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Сохраняет отчет мониторинга.

        Args:
            report: Отчет мониторинга
            output_path: Путь для сохранения
        """
        try:
            output_file = (
                Path(output_path)
                / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Отчет мониторинга сохранен: {output_file}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении отчета: {e}")

    def send_alerts(self, report: Dict[str, Any]) -> None:
        """
        Отправляет алерты при обнаружении проблем.

        Args:
            report: Отчет мониторинга
        """
        if not report.get("alerts"):
            return

        logger.info(f"Отправка {len(report['alerts'])} алертов...")

        # Здесь можно добавить отправку email, Slack уведомлений и т.д.
        for alert in report["alerts"]:
            logger.warning(
                f"ALERT [{alert['severity'].upper()}]: {alert['message']} - {alert['details']}"
            )


def main():
    """Основная функция для запуска мониторинга."""
    # Параметры (в реальном проекте они должны быть в конфигурационном файле)
    model_path = "models/trained/best_model.pkl"
    reference_data_path = "data/processed/X_train.csv"
    current_data_path = "data/processed/X_test.csv"
    test_labels_path = "data/processed/y_test.csv"

    # Проверяем наличие файлов
    required_files = [
        model_path,
        reference_data_path,
        current_data_path,
        test_labels_path,
    ]
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Файл не найден: {file_path}")
            return 1

    try:
        # Загружаем данные
        current_data = pd.read_csv(current_data_path)
        X_test = current_data
        y_test = pd.read_csv(test_labels_path).squeeze()

        # Определяем признаки
        feature_columns = X_test.columns.tolist()
        sensitive_attributes = [
            "grade",
            "emp_length",
        ]  # Пример чувствительных атрибутов

        # Создаем монитор
        monitor = ModelMonitor(
            model_path=model_path, reference_data_path=reference_data_path
        )

        # Генерируем отчет
        report = monitor.generate_monitoring_report(
            current_data=current_data,
            X_test=X_test,
            y_test=y_test,
            feature_columns=feature_columns,
            sensitive_attributes=sensitive_attributes,
        )

        # Сохраняем отчет
        monitor.save_monitoring_report(report, "monitoring/reports")

        # Отправляем алерты
        monitor.send_alerts(report)

        # Выводим краткую сводку
        print(f"\n📊 ОТЧЕТ МОНИТОРИНГА МОДЕЛИ")
        print(f"Время: {report['timestamp']}")
        print(f"Статус: {report['overall_status']}")
        print(f"Алертов: {len(report.get('alerts', []))}")

        if report.get("alerts"):
            print("\n🚨 АЛЕРТЫ:")
            for alert in report["alerts"]:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")

        return 0 if report["overall_status"] == "healthy" else 1

    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
