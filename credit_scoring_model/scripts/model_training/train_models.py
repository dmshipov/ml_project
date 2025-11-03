"""
Скрипт для обучения моделей кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку обработанных данных
2. Обучение различных моделей машинного обучения
3. Оценку качества моделей
4. Сохранение обученных моделей
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

# Импорты для визуализации
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split  # Добавлен StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Импорты для баланса классов и SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Импорт XGBoost
import xgboost as xgb

# Импорт MLflow (опциональный)
try:
    from .simple_mlflow_tracking import setup_mlflow_experiment
except ImportError:
    try:
        from simple_mlflow_tracking import setup_mlflow_experiment
    except ImportError:
        setup_mlflow_experiment = None

warnings.filterwarnings("ignore")

# Добавлено для исправления UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')


def load_processed_data(
    data_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Загружает обработанные данные (без preprocessor, чтобы избежать утечки).

    Args:
        data_dir: Папка с обработанными данными

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_dir)

    print("Загрузка обработанных данных...")

    # Загружаем данные
    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # Приводим y к int
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print(f"Загружено:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def create_models(preprocessor: ColumnTransformer) -> Dict[str, ImbPipeline]:
    """
    Создает словарь моделей для обучения, используя переданный препроцессор (с SMOTE для баланса).

    Args:
        preprocessor: Новый препроцессор (созданный на train)

    Returns:
        Dict[str, Pipeline]: Словарь с моделями (с SMOTE)
    """
    print("\nСоздание моделей...")

    # Создаем модели с балансом классов и SMOTE
    models = {
        "Logistic Regression": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "Random Forest": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight="balanced")),
        ]),
        "XGBoost": ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric="auc",
                scale_pos_weight=None,  # Установим динамически в main
            )),
        ]),
    }

    print(f"Создано {len(models)} моделей с балансом:")
    for name in models.keys():
        print(f"  - {name}")

    return models


def train_and_evaluate_models(
    models: Dict[str, ImbPipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_mlflow: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Обучает модели и оценивает их качество (с StratifiedKFold CV для имбаланса).

    Args:
        models: Словарь с моделями
        X_train: Обучающие признаки
        X_test: Тестовые признаки
        y_train: Обучающая целевая переменная
        y_test: Тестовая целевая переменная
        use_mlflow: Использовать ли MLflow

    Returns:
        Dict: Результаты обучения и оценки
    """
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 60)

    results = {}
    predictions = {}
    probabilities = {}

    # Настройка MLflow
    tracker = None
    if use_mlflow and setup_mlflow_experiment is not None:
        tracker = setup_mlflow_experiment("credit-scoring-training")

    # Обучаем каждую модель
    for name, model in models.items():
        print(f"\nОбучение {name}...")

        try:
            # Обучение
            model.fit(X_train, y_train)

            # Предсказания
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Сохраняем
            predictions[name] = y_pred
            probabilities[name] = y_proba

            # Метрики
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            # Кросс-валидация с StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            metrics["cv_auc_mean"] = cv_scores.mean()
            metrics["cv_auc_std"] = cv_scores.std()

            results[name] = metrics

            print(f"{name} завершён:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(
                f"  CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}"
            )

            # Логируем в MLflow
            if tracker is not None:
                try:
                    with tracker.start_run(run_name=name) as run:
                        tracker.log_data_info(X_train, X_test, y_train, y_test)
                        model_params = model.get_params()
                        tracker.log_model_params(model_params)
                        tracker.log_metrics(metrics)
                        tracker.log_model(model, name)
                        print(f"  MLflow run ID: {run.info.run_id}")
                except Exception as mlflow_error:
                    print(f"  Ошибка MLflow для {name}: {mlflow_error}")

        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")
            results[name] = {"error": str(e)}

    return {
        "results": results,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def create_comparison_plot(
    results: Dict[str, Dict[str, Any]], output_dir: str = "models/artifacts"
) -> None:
    
    print("\nСоздание графика сравнения моделей...")

    model_names = []
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_values = {metric: [] for metric in metrics}

    for model_name, model_results in results.items():
        if "error" not in model_results:
            model_names.append(model_name)
            for metric in metrics:
                metric_values[metric].append(model_results[metric])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_names, metric_values[metric])
        ax.set_title(f"{metric.upper()}")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars, metric_values[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    axes[5].remove()

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"График сохранен: {output_path / 'model_comparison.png'}")


def create_roc_curves(
    probabilities: Dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: str = "models/artifacts"
) -> None:

    print("\nСоздание ROC-кривых...")

    plt.figure(figsize=(10, 8))

    for name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые: Сравнение моделей")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"ROC-кривые сохранены: {output_path / 'roc_curves.png'}")


def save_models(
    models: Dict[str, ImbPipeline],
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "models/trained",
) -> None:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nСохранение моделей в {output_path}...")

    for name, model in models.items():
        if "error" not in results.get(name, {}):
            model_path = output_path / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)
            print(f"  {name} -> {model_path}")

    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_path / "model_results.csv")
    print(f"  Результаты -> {output_path / 'model_results.csv'}")

    if not results_df.empty:
        best_model_name = results_df["roc_auc"].idxmax()
        best_model = models[best_model_name]
        joblib.dump(best_model, output_path / "best_model.pkl")
        print(
            f"  Лучшая модель ({best_model_name}) -> {output_path / 'best_model.pkl'}"
        )


def print_final_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Определено num_cols/cat_cols в main""" 
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    results_df = pd.DataFrame(results).T

    if results_df.empty:
        print("Нет результатов для отображения.")
        return

    results_df = results_df.drop(columns=["error"], errors="ignore")
    results_df = results_df.sort_values("roc_auc", ascending=False)

    print("\nСравнение моделей:")
    print(results_df.round(4))

    if "roc_auc" in results_df.columns:
        best_model = results_df["roc_auc"].idxmax()
        best_auc = results_df.loc[best_model, "roc_auc"]
        print(f"\nЛучшая модель: {best_model} (ROC-AUC: {best_auc:.4f})")


def main():
    """Основная функция с исправлениями: создание preprocessor, проверки утечки, динамическая настройка XGBoost.""" 
    # Загружаем данные
    X_train, X_test, y_train, y_test = load_processed_data()

    print("\nПроверки на утечку данных:")
    print("Пропорции классов y_train:")
    print(y_train.value_counts(normalize=True))
    print("Пропорции классов y_test:")
    print(y_test.value_counts(normalize=True))

    print(f"Размеры: X_train {X_train.shape}, X_test {X_test.shape}")
    if X_train.shape == X_test.shape:
        print("Размеры одинаковы — проверка на идентичность...")
        if X_train.equals(X_test):
            print("ОШИБКА: X_train и X_test идентичны — утечка данных!")
            return
        else:
            print("Идентичность не найдена.")
    else:
        print("Размеры разные — нормально.")

    print("\nПроверка корреляций (top 5 с target):")
    high_corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    print(high_corr.head(5))
    if high_corr.max() > 0.95:
        print("ВНИМАНИЕ: Очень высокая корреляция — возможная утечка через признаки!")

    # Определяем колонки для preprocessor
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Создаем preprocessor на train
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Создаем модели
    models = create_models(preprocessor)

    # Динамическая настройка scale_pos_weight для XGBoost
    neg_pos_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    models["XGBoost"].named_steps["classifier"].scale_pos_weight = neg_pos_ratio

    # Обучаем и оцениваем
    training_results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

    # Создаем визуализации
    create_comparison_plot(training_results["results"])
    create_roc_curves(training_results["probabilities"], y_test)

    # Сохраняем модели
    save_models(models, training_results["results"])

    # Выводим финальные результаты
    print_final_results(training_results["results"])

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО УСПЕШНО")
    print("=" * 60)


if __name__ == "__main__":
    main()