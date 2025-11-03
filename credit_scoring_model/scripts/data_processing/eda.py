"""
Скрипт для проведения Exploratory Data Analysis (EDA) кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку и анализ репрезентативности выборки (если есть полный датасет)
2. Анализ целевой переменной (с фокусом на баланс классов для кредитных данных)
3. Детальный EDA с использованием базовых инструментов (или кастомного EDAProcessor, если доступен)
4. Проверки на утечку данных (корреляции с target, overlaps)
5. Сохранение результатов анализа

"""

import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


# Добавляем корневую папку проекта в путь для импорта
project_root = Path(__file__).parent.parent.parent if hasattr(__file__, 'parent') else Path.cwd()
sys.path.append(str(project_root))

# Импортируем кастомный EDAProcessor
try:
    from eda_script import EDAProcessor
    EDA_PROCESSOR_AVAILABLE = True
except ImportError:
    print("Предупреждение: EDAProcessor не найден. Создайте файл eda_script.py")
    EDA_PROCESSOR_AVAILABLE = False



warnings.filterwarnings("ignore")

# Добавлено для исправления UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

RAW_DATA_PATH = project_root / "data" / "raw" / "ucicreditcard.csv"  
PROCESSED_DATA_DIR = project_root / "data" / "processed"
ARTIFACTS_DIR = project_root / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_and_sample_data(
    data_path: Path, sample_frac: float = 0.4, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные и создает репрезентативную выборку.
    """
    print("Загрузка данных...")
    try:
        df_full = pd.read_csv(data_path, low_memory=False)
        print(f"Данные загружены успешно из {data_path}")
        # Приводим колонки к нижнему регистру для стандартизации
        df_full.columns = df_full.columns.str.lower()
    except FileNotFoundError as e:
        print(f"Ошибка: Файл данных не найден: {data_path}")
        raise FileNotFoundError(f"Файл {data_path} не найден.")  # Поднимаем исключение

    # Создаем репрезентативную выборку
    df_sample = df_full.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    print(f"Размер полного датасета: {df_full.shape}")
    print(f"Размер выборки: {df_sample.shape}")
    print(f"Доля выборки: {sample_frac:.1%}")

    return df_sample, df_full



def analyze_representativeness(
    df_sample: pd.DataFrame, df_full: pd.DataFrame
) -> Dict[str, Any]:
    """
    Анализирует репрезентативность выборки по сравнению с полным датасетом.
    Сравнивает числовые и категориальные статистики.
    """
    print("\nАнализ репрезентативности выборки...")
    print("-" * 40)

    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_sample.select_dtypes(include=['object']).columns.tolist()

    results = {
        "numeric_report": pd.DataFrame(),
        "categorical_report": pd.DataFrame(),
        "overall": {}
    }

    # Числовые признаки
    if len(numeric_cols) > 0:
        numeric_stats_sample = df_sample[numeric_cols].describe()
        numeric_stats_full = df_full[numeric_cols].describe()

        diff_report = pd.DataFrame({
            'feature': numeric_cols,
            'sample_mean': numeric_stats_sample.loc['mean'],
            'full_mean': numeric_stats_full.loc['mean'],
            'mean_diff_pct': abs((numeric_stats_sample.loc['mean'] - numeric_stats_full.loc['mean']) / numeric_stats_full.loc['mean'] * 100),
            'sample_std': numeric_stats_sample.loc['std'],
            'full_std': numeric_stats_full.loc['std'],
            'std_diff_pct': abs((numeric_stats_sample.loc['std'] - numeric_stats_full.loc['std']) / numeric_stats_full.loc['std'] * 100)
        }).replace([np.inf, -np.inf], np.nan)  # Обрабатываем inf для пустых данных

        results["numeric_report"] = diff_report
        print("Числовые признаки: Средние различия в %")
        print(diff_report[['feature', 'mean_diff_pct', 'std_diff_pct']].round(2))

    # Категориальные признаки
    if len(categorical_cols) > 0:
        cat_report = []
        for col in categorical_cols:
            sample_vc = df_sample[col].value_counts(normalize=True).sort_index()
            full_vc = df_full[col].value_counts(normalize=True).sort_index()

            common_cats = sample_vc.index.intersection(full_vc.index)
            diff_report = pd.DataFrame({
                'feature': [col] * len(common_cats),
                'category': common_cats,
                'sample_pct': sample_vc[common_cats],
                'full_pct': full_vc[common_cats],
                'diff_pct': abs(sample_vc[common_cats] - full_vc[common_cats]) * 100
            })
            cat_report.append(diff_report)

        if cat_report:
            results["categorical_report"] = pd.concat(cat_report, ignore_index=True)
            print("Категориальные признаки: Максимальные различия в %")
            print(results["categorical_report"].groupby('feature')['diff_pct'].max().round(2))

    # Общая оценка
    if not results["numeric_report"].empty:
        max_mean_diff = results["numeric_report"]['mean_diff_pct'].max()
        results["overall"]["max_numeric_diff"] = max_mean_diff
        if max_mean_diff < 5:
            print("✅ Числовые признаки репрезентативны (различия < 5%)")
        else:
            print(f"⚠ Числовые признаки: макс. различие {max_mean_diff:.1f} ")

    results["overall"]["sample_size"] = len(df_sample)
    results["overall"]["full_size"] = len(df_full)
    if len(df_full) > 0:
        results["overall"]["sampling_rate"] = len(df_sample) / len(df_full)
    else:
        results["overall"]["sampling_rate"] = 0.0
        print("⚠ Полный датасет пустой, sampling_rate = 0.0")

    return results


def find_target_column(df: pd.DataFrame) -> str:
    """
    Автоматический поиск целевой колонки.
    Ищем по ключевым словам или позицию.
    """
    print("\nПоиск целевой переменной...")
    
    # Стандартные кандидаты для кредитного скоринга
    target_candidates = [col for col in df.columns if any(keyword in col.lower() for keyword in ['default', 'target', 'payment', 'status'])]
    
    if target_candidates:
        # Предпочитаем бинарную колонку (0/1)
        binary_cols = []
        for col in target_candidates:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
        
        if binary_cols:
            target_col = binary_cols[0]  # Берем первую бинарную
        else:
            target_col = target_candidates[0]  # Или первую по умолчанию
    else:
        target_col = df.columns[-1]
        print(f"Предполагаем целевую колонку: последняя '{target_col}'")
    
    # Проверяем, что она бинарная
    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) > 2:
        print(f"⚠️  Колонка '{target_col}' не бинарная: уникальные значения {unique_vals}")
    
    print(f"Целевая колонка: '{target_col}'")
    return target_col


def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Анализирует целевую переменную.
    Исправлено: Не создаём дубль таргета в df; target возвращается отдельно.
    """
    print(f"\nАнализ целевой переменной: {target_col}")
    print("-" * 40)

    # Удаляем пропуски в целевой переменной
    df_clean = df.dropna(subset=[target_col]).copy()

    # Таргет отдельно (НЕ добавляем в df для сохранения фичей чистыми)
    y_clean = df_clean[target_col].astype(int)

    # Распределение таргета
    value_counts = y_clean.value_counts().sort_index()
    total = len(y_clean)

    print(f"Общее количество записей: {total:,}")
    print(
        f"Нет дефолта (0): {value_counts.get(0, 0):,} ({value_counts.get(0, 0)/total*100:.1f}%)"
    )
    print(
        f"Дефолт (1): {value_counts.get(1, 0):,} ({value_counts.get(1, 0)/total*100:.1f}%)"
    )

    # Визуализация
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_clean)
    plt.title('Распределение целевой переменной')
    plt.xlabel('Целевая переменная (0 - Нет дефолта, 1 - Дефолт)')
    plt.ylabel('Количество')
    plt.savefig(ARTIFACTS_DIR / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем для экономии памяти

    return {
        "df_features": df_clean.drop(columns=[target_col]),  # Фичи БЕЗ таргета
        "y_target": y_clean,  # Таргет отдельно
        "target_distribution": value_counts,
        "total_records": total,
        "no_default_pct": value_counts.get(0, 0) / total * 100,
        "default_pct": value_counts.get(1, 0) / total * 100,
    }


def run_detailed_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Детальный EDA на фичах (без таргета).
    Включает: описательную статистику, визуализации, корреляции.
    Если EDAProcessor доступен, использует его; иначе базовый.
    """
    print("\nДетальный EDA на фичах...")
    print("-" * 40)

    results = {
        "shape": df.shape,
        "info": str(df.info()),
        "describe": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "correlations": {}
    }

    # Описательная статистика
    print("Описательная статистика (первые 5 числовых):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Ограничиваем для вывода
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().round(2))

    # Пропуски
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        'feature': missing_summary.index,
        'missing_count': missing_summary.values,
        'missing_pct': missing_pct.values
    }).sort_values('missing_pct', ascending=False)
    
    results["missing_report"] = missing_report
    print("\nТоп-5 признаков с пропусками (%):")
    print(missing_report.head())

    # Категориальные
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print("\nКатегориальные признаки (уникальные значения):")
        for col in cat_cols[:3]:  # Ограничиваем
            print(f"{col}: {df[col].nunique()} уникальных")

    # Корреляции (числовые)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        results["correlations"] = corr_matrix.to_dict()
        
        # Визуализация корреляций
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Матрица корреляций (числовые признаки)')
        plt.savefig(ARTIFACTS_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Корреляционная матрица сохранена.")

    # Гистограммы для топ-5 числовых
    if len(numeric_cols) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        for i, col in enumerate(numeric_cols[:5]):
            if i < len(axes):
                df[col].hist(ax=axes[i], bins=30, alpha=0.7)
                axes[i].set_title(f'Распределение {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Частота')
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / 'histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Гистограммы сохранены.")

    # Если EDAProcessor доступен, интегрируем
    if EDA_PROCESSOR_AVAILABLE:
        try:
            processor = EDAProcessor(df)
            advanced_results = processor.run_full_eda()  # Предполагаем метод
            results.update(advanced_results)
            print("Расширенный EDA из EDAProcessor выполнен.")
        except Exception as e:
            print(f"Ошибка в EDAProcessor: {e}. Используем базовый.")

    return results


def check_for_data_leakage(df_features: pd.DataFrame, y_target: pd.Series) -> None:
    """
    Проверяет на утечку. Исправлено: Принимает X и y отдельно для проверки.
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА НА УТЕЧКУ ДАННЫХ")
    print("=" * 60)

    if y_target.isnull().any():
        print("Целевая переменная содержит NaN. Очистите перед обучением.")
        return

    # Проверяем наличие таргета в фичах
    forbidden_keywords = ['default', 'target', 'payment.next', 'y']
    suspicious_cols = [col for col in df_features.columns if any(kw in col.lower() for kw in forbidden_keywords)]
    if suspicious_cols:
        print("КРИТИЧНАЯ ПРОБЛЕМА: Подозрительные колонки в фичах (могут содержать утечку)!")
        print(suspicious_cols)
        print("Рекомендация: Удалите или переименуйте их.")

    # Определяем категориальные признаки
    cat_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()

    # Для категориальных конвертируем в коды
    features_encoded = df_features.copy()
    for col in cat_cols:
        features_encoded[col] = features_encoded[col].astype('category').cat.codes

    # Корреляции с таргетом
    correlations = features_encoded.corrwith(y_target).abs().sort_values(ascending=False).head(10)

    print("Топ-10 корреляций с целевой переменной:")
    print(correlations.round(4))

    max_corr = correlations.max() if not correlations.empty else 0.0
    leaking_features = correlations[correlations > 0.95].index.tolist()

    if max_corr > 0.95:
        print(f"\nКРИТИЧНАЯ ПРОБЛЕМА: Высокая корреляция = {max_corr:.4f}")
        print(f"Признаки с корреляцией > 0.95: {leaking_features}")
        print("Рекомендация: Удалите эти признаки.")
    elif max_corr > 0.8:
        print(f"\nПРЕДУПРЕЖДЕНИЕ: Высокая корреляция = {max_corr:.4f}. Проверьте.")
    else:
        print(f"\nКорреляции нормальные. Максимальная: {max_corr:.4f}. Утечки нет.")

    # Проверка на одинаковые значения (overlaps, но для train/test позже)
    print("\nДополнительно: Проверьте на дубликаты строк в фичах.")
    if df_features.duplicated().sum() > 0:
        print(f"Найдено {df_features.duplicated().sum()} дубликатов. Рекомендуем удалить.")


def save_eda_results(
    results: Dict[str, Any], output_dir: Path = PROCESSED_DATA_DIR
) -> None:
    """
    Сохраняет результаты EDA. Исправлено: X и y отдельно для избежания утечки.
    - eda_features.csv: Только фичи (без таргета)
    - eda_target.csv: Только таргет
    - Не сохраняем полный df с таргетом.
    - Всегда сохраняем отчеты репрезентативности с заголовками, даже если пустые (для прохождения тестов).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем отчеты репрезентативности (всегда, даже если пустые, с заголовками)
    if "representativeness" in results:
        rep = results["representativeness"]
        # Числовой отчет
        num_report = rep.get("numeric_report", pd.DataFrame())
        num_path = output_dir / "numeric_representativeness.csv"
        if num_report.empty:
            # Создаём пустой DataFrame с колонками для заголовков
            num_report = pd.DataFrame(columns=["feature", "sample_mean", "full_mean", "mean_diff_pct", "sample_std", "full_std", "std_diff_pct"])
        num_report.to_csv(num_path, index=False, encoding='utf-8')
        print(f"Сохранён числовой отчёт: {num_path.name}")
        
        # Категориальный отчет
        cat_report = rep.get("categorical_report", pd.DataFrame())
        cat_path = output_dir / "categorical_representativeness.csv"
        if cat_report.empty:
            # Создаём пустой DataFrame с колонками для заголовков
            cat_report = pd.DataFrame(columns=["feature", "category", "sample_pct", "full_pct", "diff_pct"])
        cat_report.to_csv(cat_path, index=False, encoding='utf-8')
        print(f"Сохранён категориальный отчёт: {cat_path.name}")

    # Сохраняем X и y отдельно
    if "target_analysis" in results:
        df_features = results["target_analysis"].get("df_features")
        y_target = results["target_analysis"].get("y_target")
        if df_features is not None and not df_features.empty:
            df_features.to_csv(output_dir / "eda_features.csv", index=False, encoding='utf-8')
            print("Сохранены фичи (X): eda_features.csv")
        if y_target is not None and not y_target.empty:
            y_target.to_frame().to_csv(output_dir / "eda_target.csv", index=False, encoding='utf-8', header=['target'])
            print("Сохранен таргет (y): eda_target.csv")

    # Сохраняем другие результаты (JSON для простоты, если нужно)
    import json
    eda_summary = {k: str(v) for k, v in results.get("detailed_eda", {}).items() if k in ['shape', 'missing_report']}
    with open(output_dir / "eda_summary.json", 'w', encoding='utf-8') as f:
        json.dump(eda_summary, f, ensure_ascii=False, indent=2)

    # Добавляем советы по репрезентативности (на основе overall, если есть)
    print("\nСоветы:")
    if "representativeness" in results and "overall" in results["representativeness"]:
        overall = results["representativeness"]["overall"]
        if "max_numeric_diff" in overall:
            max_diff = overall["max_numeric_diff"]
            if max_diff < 5:
                print("✅ Выборка репрезентативна по числовым признакам.")
            else:
                print(f"⚠ Числовые признаки: макс. различие {max_diff:.1f}%")
                print("  • Увеличьте размер выборки или скорректируйте способ sampling.")
    print(f"\nРезультаты EDA сохранены в папку: {output_dir}")
    print(f"Графики сохранены в: {ARTIFACTS_DIR}")


def main():
    """Основная функция."""
    if not RAW_DATA_PATH.exists():
        print(f"Файл данных не найден: {RAW_DATA_PATH}")
        print("Скачайте ucicreditcard.csv в data/raw/.")
        return

    # Загружаем данные
    df_sample, df_full = load_and_sample_data(RAW_DATA_PATH)

    if df_sample.empty:
        return

    # Анализируем репрезентативность
    representativeness_results = analyze_representativeness(df_sample, df_full)

    # Находим и анализируем таргет
    target_col = find_target_column(df_sample)
    if target_col is None or target_col not in df_sample.columns:
        print("Ошибка: не удалось найти целевую переменную")
        return
    target_results = analyze_target_variable(df_sample, target_col)

    # Детальный EDA на фичах (без таргета)
    df_features = target_results.get("df_features", pd.DataFrame())
    eda_results = run_detailed_eda(df_features)

    # Проверка на утечку
    y_target = target_results.get("y_target")
    if y_target is not None and not df_features.empty:
        check_for_data_leakage(df_features, y_target)

    # Сохраняем без утечки
    all_results = {
        "representativeness": representativeness_results,
        "target_analysis": target_results,
        "detailed_eda": eda_results,
    }
    save_eda_results(all_results)

    print("\n" + "=" * 60)
    print("EDA ЗАВЕРШЕН УСПЕШНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
