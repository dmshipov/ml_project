"""
Скрипт для подбора гиперпараметров моделей кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку обработанных данных (сырых, так как препроцессор интегрируется в Pipeline)
2. Определение сеток параметров для различных моделей (упрощенные для скорости)
3. Поиск лучших гиперпараметров с помощью GridSearchCV
4. Оценку качества настроенных моделей
5. Сохранение лучших моделей и результатов
6. Визуализацию анализа гиперпараметров и сравнения моделей
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
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
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Фиксируем кодировку для корректного вывода на Windows
sys.stdout.reconfigure(encoding='utf-8')


def load_processed_data(
    data_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Загружает сырые обработанные данные из eda_features.csv (фичи) и eda_target.csv (таргет),
    затем делит на train/test с помощью train_test_split (test_size=0.2, stratify по y для баланса классов).
    Это обеспечивает совместимость с обновленным EDA, где фичи и таргет разделены, и предотвращает утечку данных.

    Args:
        data_dir: Папка с данными

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_dir)

    print("Загрузка сырых обработанных данных из EDA...")

    try:
        # Загрузка объединенного датасета 
        X_full = pd.read_csv(data_path / "eda_features.csv")
        y_full = pd.read_csv(data_path / "eda_target.csv")['target'].squeeze()  # Предполагаем колонку 'target'

        # Разделение на train/test с стратификацией для баланса классов
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        print(f"Загружено и разбито:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")

        # Диагностика на утечку данных
        print("\nДиагностика на утечку данных:")
        print("  - Проверка, не входит ли таргет в фичи:")
        print(f"    Цель в X_train? {'default.payment.next.month' in X_train.columns or 'y' in X_train.columns or 'target' in X_train.columns}")
        print("  - Корреляция y_train с X_train (|corr| > 0.95):")
        correlations = X_train.corrwith(y_train)
        high_corr = correlations[abs(correlations) > 0.95]
        if not high_corr.empty:
            print(f"    Высокие корреляции найдены:\n{high_corr}")
        else:
            print("    Нет фич с высокой корреляцией (>0.95).")

        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Ошибка: Файл данных не найден: {e}")
        print("Убедитесь, что eda_features.csv и eda_target.csv существуют в data/processed/ после запуска EDA.")
        raise


def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Создает препроцессор для обработки признаков.

    Args:
        X_train: Обучающие данные для определения типов признаков

    Returns:
        ColumnTransformer: Настроенный препроцессор
    """
    print("\nСоздание препроцессора...")

    # Определяем типы признаков
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    print(
        f"Препроцессор создан для {len(numeric_features)} числовых и {len(categorical_features)} категориальных признаков"
    )

    return preprocessor


def define_parameter_grids() -> List[Dict[str, Any]]:
    """
    Определяет упрощенные сетки параметров для логистической регрессии и случайного леса.
    Каждая модель имеет свой dict с моделью и ее параметрами.

    Returns:
        List[Dict]: Список словарей с моделью и параметрами для GridSearchCV
    """
    print("\nОпределение упрощенных сеток параметров...")

    param_grids = [
        # Logistic Regression
        {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "params": {
                "classifier__C": [0.01, 0.1, 1],  # Добавлен 0.01 для сильнее регуляризации
                "classifier__penalty": ["l2"],
                "classifier__solver": ["liblinear"],
            }
        },
        # Random Forest
        {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [5, 10, 15],  # Добавлена глубина 5 для уменьшения оверфита
                "classifier__min_samples_split": [2, 5, 10],  # Добавлен 10 для листочков
                "classifier__min_samples_leaf": [1, 2, 4],  # Добавлено 4
            }
        },
    ]

    print(f"Создано {len(param_grids)} сеток параметров:")
    for i, grid in enumerate(param_grids):
        model_name = grid["model"].__class__.__name__
        print(f"  {i+1}. {model_name}")

    return param_grids


def perform_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    param_grids: List[Dict[str, Any]],
    cv: int = 5,  # Увеличено до 5 для более стабильной CV
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Выполняет поиск по сетке параметров для каждой модели отдельно.

    Args:
        X_train: Обучающие признаки
        y_train: Обучающая целевая переменная
        preprocessor: Препроцессор
        param_grids: Список сеток параметров
        cv: Количество фолдов для кросс-валидации
        n_jobs: Количество параллельных процессов

    Returns:
        Dict: Результаты поиска по сетке для каждой модели
    """
    print(f"\n" + "=" * 60)
    print("ПОИСК ПО СЕТКЕ ПАРАМЕТРОВ")
    print("=" * 60)

    results = {}

    for item in tqdm(param_grids, desc="Поиск параметров", unit="модель"):
        model = item["model"]
        param_grid = item["params"]
        model_name = model.__class__.__name__

        print(f"\nПоиск параметров для {model_name}...")

        try:
            # Создаем пайплайн с моделью
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model),
            ])

            # Выполняем поиск по сетке
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=n_jobs, verbose=0  # Уменьшен verbose для чистоты
            )

            # Обучаем
            grid_search.fit(X_train, y_train)

            # Сохраняем результаты
            results[model_name] = {
                "best_estimator": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "cv_results": grid_search.cv_results_,
            }

            print(f"{model_name} завершён:")
            print(f"  Лучший AUC (CV): {grid_search.best_score_:.4f}")
            print(f"  Лучшие параметры: {grid_search.best_params_}")

        except Exception as e:
            print(f"Ошибка при поиске параметров для {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results


def evaluate_tuned_models(
    results: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Оценивает качество настроенных моделей на тестовых данных.

    Args:
        results: Результаты поиска по сетке (с CV на train)
        X_test: Тестовые признаки
        y_test: Тестовая целевая переменная

    Returns:
        Dict: Результаты оценки
    """
    print(f"\n" + "=" * 60)
    print("ОЦЕНКА НАСТРОЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)

    evaluation_results = {}

    for model_name, model_results in results.items():
        if "error" in model_results:
            print(f"Пропускаем {model_name} из-за ошибки")
            continue

        print(f"\nОценка {model_name}...")

        try:
            best_model = model_results["best_estimator"]

            # Предсказания
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            # Вычисляем метрики
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            # Кросс-валидация на тестовой выборке для оценки дисперсии
            cv_scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring="roc_auc", n_jobs=-1)
            metrics["cv_auc_mean"] = cv_scores.mean()
            metrics["cv_auc_std"] = cv_scores.std()

            evaluation_results[model_name] = {
                "metrics": metrics,
                "predictions": y_pred,
                "probabilities": y_proba,
                "best_params": model_results["best_params"],
                "cv_score": model_results["best_score"],
            }

            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  CV AUC (на train): {model_results['best_score']:.4f}")
            print(f"  CV AUC (на test): {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

        except Exception as e:
            print(f"Ошибка при оценке {model_name}: {e}")
            evaluation_results[model_name] = {"error": str(e)}

    return evaluation_results


def create_hyperparameter_plots(
    results: Dict[str, Any], output_dir: str = "models/artifacts"
) -> None:
    """
    Создает графики для анализа гиперпараметров.

    Args:
        results: Результаты поиска по сетке
        output_dir: Папка для сохранения
    """
    print("\nСоздание графиков анализа гиперпараметров...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model_results in results.items():
        if "error" in model_results or "cv_results" not in model_results:
            continue

        try:
            cv_results = model_results["cv_results"]
            param_names = [key for key in cv_results.keys() if key.startswith("param_classifier")]

            if not param_names:
                continue

            fig, axes = plt.subplots((len(param_names) + 1) // 2, 2, figsize=(12, 8))
            if len(param_names) == 1:
                axes = [axes]

            # Обработка для множественных оси
            flat_axes = axes.ravel() if hasattr(axes, 'ravel') else [axes] if len(param_names) == 1 else axes.flatten()

            for i, param_name in enumerate(param_names):
                if i >= len(flat_axes):
                    break
                ax = flat_axes[i]
                param_values = cv_results[param_name]
                mean_scores = cv_results["mean_test_score"]

                # Для числовых параметров строим линию, для категориальных - бар
                if all(isinstance(v, (int, float, np.number)) for v in param_values.data if pd.notna(v)):
                    unique_vals = sorted(set(param_values.data))
                    scores = [mean_scores[param_values == v].mean() for v in unique_vals]
                    ax.plot(unique_vals, scores, "o-")
                else:
                    unique_vals = list(set(str(v) for v in param_values.data))
                    scores = [mean_scores[param_values == v].mean() for v in unique_vals]
                    ax.bar(unique_vals, scores, alpha=0.7)

                ax.set_title(f"{param_name.replace('param_classifier__', '').replace('_', ' ').title()}")
                ax.set_ylabel("CV Score")
                ax.grid(True, alpha=0.3)

            plt.suptitle(f"Анализ гиперпараметров: {model_name}", fontsize=16)
            plt.tight_layout()

            plot_path = output_path / f"hyperparameter_analysis_{model_name.lower()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()  # Закрываем, чтобы не переполнять память

            print(f"График сохранен: {plot_path}")

        except Exception as e:
            print(f"Ошибка при создании графика для {model_name}: {e}")


def create_comparison_plot(
    evaluation_results: Dict[str, Dict[str, Any]], output_dir: str = "models/artifacts"
) -> None:
    """
    Создает график сравнения настроенных моделей.

    Args:
        evaluation_results: Результаты оценки
        output_dir: Папка для сохранения
    """
    print("\nСоздание графика сравнения настроенных моделей...")

    valid_results = {k: v for k, v in evaluation_results.items() if "error" not in v and "metrics" in v}
    if not valid_results:
        print("Нет данных для сравнения.")
        return

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(valid_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [valid_results[m]["metrics"][metric] for m in model_names]
        bars = axes[i].bar(model_names, values, alpha=0.8)
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", ha="center", va="bottom")

    axes[5].remove()
    plt.suptitle("Сравнение настроенных моделей", fontsize=16)
    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "tuned_models_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"График сохранен: {plot_path}")


def save_tuned_models(
    results: Dict[str, Any],
    evaluation_results: Dict[str, Dict[str, Any]],
    output_dir: str = "models/trained",
) -> None:
    """
    Сохраняет настроенные модели и результаты.

    Args:
        results: Результаты поиска по сетке
        evaluation_results: Результаты оценки
        output_dir: Папка для сохранения
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Сохранение настроенных моделей в {output_path}...")

    # Сохраняем каждую модель
    for model_name, model_results in results.items():
        if "error" not in model_results and "best_estimator" in model_results:
            model_path = output_path / f"tuned_{model_name.lower()}.pkl"
            joblib.dump(model_results["best_estimator"], model_path)
            print(f"  {model_name} -> {model_path}")

    # Сохраняем результаты
    if evaluation_results:
        results_data = []
        for model_name, model_results in evaluation_results.items():
            if "error" not in model_results and "metrics" in model_results:
                row = {"model": model_name, **model_results["metrics"], **model_results["best_params"]}
                results_data.append(row)

        if results_data:
            results_df = pd.DataFrame(results_data)
            csv_path = output_path / "tuned_models_results.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"  Результаты -> {csv_path}")

    # Сохраняем лучшую модель
    best_model_name = max(
        [name for name in evaluation_results if "error" not in evaluation_results[name]],
        key=lambda x: evaluation_results[x]["metrics"].get("roc_auc", 0),
        default=None
    )
    if best_model_name and "best_estimator" in results.get(best_model_name, {}):
        best_model = results[best_model_name]["best_estimator"]
        best_path = output_path / "best_tuned_model.pkl"
        joblib.dump(best_model, best_path)
        print(f"  Лучшая модель ({best_model_name}) -> {best_path}")


def print_final_results(evaluation_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Выводит финальные результаты сравнения настроенных моделей.

    Args:
        evaluation_results: Результаты оценки
    """
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НАСТРОЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)

    if not evaluation_results:
        print("Нет результатов для отображения.")
        return

    results_data = []
    for model_name, model_results in evaluation_results.items():
        if "error" not in model_results and "metrics" in model_results:
            results_data.append({"model": model_name, **model_results["metrics"]})

    if not results_data:
        print("Нет валидных результатов.")
        return

    results_df = pd.DataFrame(results_data).sort_values("roc_auc", ascending=False)
    print("\nСравнение настроенных моделей:")
    print(results_df.round(4))

    best_row = results_df.iloc[0]
    print(f"\n🏆 Лучшая модель: {best_row['model']} (ROC-AUC: {best_row['roc_auc']:.4f})")


def main():
    """Основная функция для запуска подбора гиперпараметров."""
    print("🚀 ЗАПУСК ПОДБОРА ГИПЕРПАРАМЕТРОВ С GRIDSEARCHCV")
    print("=" * 60)

    try:
        # 1. Загружаем и разбиваем данные
        X_train, X_test, y_train, y_test = load_processed_data()

        # 2. Создаем препроцессор
        preprocessor = create_preprocessor(X_train)

        # 3. Определяем сетки параметров
        param_grids = define_parameter_grids()

        # 4. Выполняем поиск
        grid_search_results = perform_grid_search(X_train, y_train, preprocessor, param_grids, cv=5)

        # 5. Оцениваем
        evaluation_results = evaluate_tuned_models(grid_search_results, X_test, y_test)

        # 6. Визуализации
        create_hyperparameter_plots(grid_search_results)
        create_comparison_plot(evaluation_results)

        # 7. Сохранение
        save_tuned_models(grid_search_results, evaluation_results)

        # 8. Финальные результаты
        print_final_results(evaluation_results)

        print("\n" + "=" * 60)
        print("✅ ПОДБОР ГИПЕРПАРАМЕТРОВ ЗАВЕРШЕН УСПЕШНО")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
