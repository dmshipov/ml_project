"""
Unit тесты для модуля EDA (eda.py).
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_processing.eda import (
    analyze_representativeness,
    analyze_target_variable,
    check_for_data_leakage,
    find_target_column,
    load_and_sample_data,
    run_detailed_eda,
    save_eda_results,
)


class TestEDA:
    """Тесты для функций EDA."""

    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            "id": range(1000),
            "limit_bal": np.random.normal(150000, 50000, 1000),
            "sex": np.random.choice(["Male", "Female"], 1000),
            "education": np.random.choice(["High School", "Bachelor", "Master"], 1000),
            "marriage": np.random.choice(["Married", "Single"], 1000),
            "age": np.random.normal(35, 10, 1000),
            "pay_0": np.random.normal(0, 1, 1000),
            "bill_amt1": np.random.normal(50000, 20000, 1000),
            "pay_amt1": np.random.normal(2000, 1000, 1000),
            "default.payment.next.month": np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Создает временный CSV файл."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_dir(self):
        """Создает временную директорию."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def test_load_and_sample_data(self, temp_csv_file):
        """Тест загрузки и выборки данных."""
        sample_df, full_df = load_and_sample_data(
            temp_csv_file, sample_frac=0.2, random_state=42
        )

        assert isinstance(sample_df, pd.DataFrame)
        assert isinstance(full_df, pd.DataFrame)
        assert len(sample_df) == 200  # 20% от 1000, подвыборка
        assert len(full_df) == 1000  # Полный датасет
        assert sample_df.shape[1] == full_df.shape[1]

        # Проверяем, что колонки в нижнем регистре
        assert all(col.islower() for col in sample_df.columns)

    def test_load_and_sample_data_invalid_path(self):
        """Тест загрузки с неверным путем."""
        with pytest.raises(FileNotFoundError):
            load_and_sample_data("nonexistent_file.csv")

    def test_analyze_representativeness(self, sample_data):
        """Тест анализа репрезентативности."""
        # Создаем подвыборку
        df_sample = sample_data.sample(frac=0.2, random_state=42)

        result = analyze_representativeness(df_sample, sample_data)

        assert isinstance(result, dict)
        assert "numeric_report" in result
        assert "categorical_report" in result
        assert "overall" in result

        # Проверяем структуру отчета
        assert isinstance(result["numeric_report"], pd.DataFrame)
        assert isinstance(result["categorical_report"], pd.DataFrame)
        assert isinstance(result["overall"], dict)
        assert "max_numeric_diff" in result["overall"]
        assert "sample_size" in result["overall"]
        assert "full_size" in result["overall"]
        assert "sampling_rate" in result["overall"]

    def test_find_target_column(self, sample_data):
        """Тест поиска целевой переменной."""
        # Тест с default.payment.next.month
        target_col = find_target_column(sample_data)
        assert target_col == "default.payment.next.month"

        # Тест с переименованной колонкой
        sample_data_renamed = sample_data.rename(columns={"default.payment.next.month": "target"})
        target_col = find_target_column(sample_data_renamed)
        assert target_col == "target"

        # Тест без целевой переменной
        sample_data_no_target = sample_data.drop(columns=["default.payment.next.month"])
        target_col = find_target_column(sample_data_no_target)
        assert target_col == sample_data_no_target.columns[-1]  # Возвращает последнюю колонку

    def test_analyze_target_variable(self, sample_data):
        """Тест анализа целевой переменной."""
        result = analyze_target_variable(sample_data, "default.payment.next.month")

        assert isinstance(result, dict)
        keys = [
            "df_features",
            "y_target",
            "target_distribution",
            "total_records",
            "no_default_pct",
            "default_pct",
        ]
        for key in keys:
            assert key in result

        # Проверяем, что данные очищены
        assert isinstance(result["df_features"], pd.DataFrame)
        assert "default.payment.next.month" not in result["df_features"].columns  # df_features без таргета

        # Проверяем таргет
        assert isinstance(result["y_target"], pd.Series)
        assert result["y_target"].dtype == int

        # Проверяем распределение
        assert isinstance(result["target_distribution"], pd.Series)
        assert result["total_records"] > 0
        assert 0 <= result["no_default_pct"] <= 100
        assert 0 <= result["default_pct"] <= 100

    def test_analyze_target_variable_with_nulls(self):
        """Тест анализа целевой переменной с пропусками."""
        data_with_nulls = pd.DataFrame(
            {
                "default.payment.next.month": [0, 1, None, 0, 1],
                "feature1": [1, 2, 3, 4, 5],
            }
        )

        result = analyze_target_variable(data_with_nulls, "default.payment.next.month")

        assert isinstance(result, dict)
        # Проверяем, что пропуски удалены из таргета и фичей
        assert result["df_features"].shape[0] == 4  # Удалены строки с NaN в таргете
        assert len(result["y_target"]) == 4
        assert result["y_target"].notna().all()

    def test_run_detailed_eda(self, sample_data):
        """Тест детального EDA."""
        result = run_detailed_eda(sample_data.drop(columns=["default.payment.next.month"]))

        assert isinstance(result, dict)
        assert "shape" in result
        assert "info" in result
        assert "describe" in result
        assert "missing_values" in result
        assert "correlations" in result
        assert "missing_report" in result

    @patch("builtins.print")
    def test_check_for_data_leakage(self, mock_print, sample_data):
        """Тест проверки на утечку данных."""
        df_features = sample_data.drop(columns=["default.payment.next.month"])
        y_target = sample_data["default.payment.next.month"].astype(int)

        check_for_data_leakage(df_features, y_target)

        # Проверяем, что print был вызван (для логики утечки сложно проверять сообщения напрямую без сложного мока)
        assert mock_print.called

    def test_save_eda_results(self, temp_dir, sample_data):
        """Тест сохранения результатов EDA."""
        results = {
            "representativeness": {
                "numeric_report": pd.DataFrame({
                    'feature': ['age'],
                    'sample_mean': [35.0],
                    'full_mean': [36.0],
                    'mean_diff_pct': [2.86],
                    'sample_std': [10.0],
                    'full_std': [10.5],
                    'std_diff_pct': [5.0],
                }),
                "categorical_report": pd.DataFrame({  # Добавлено для гарантии создания файла
                    'feature': [],
                    'category': [],
                    'sample_pct': [],
                    'full_pct': [],
                    'diff_pct': [],
                }),
                "overall": {"max_numeric_diff": 2.86, "sample_size": 200, "full_size": 1000, "sampling_rate": 0.2}
            },
            "target_analysis": {
                "df_features": sample_data.drop(columns=["default.payment.next.month"]),
                "y_target": sample_data["default.payment.next.month"].astype(int),
                "target_distribution": sample_data["default.payment.next.month"].value_counts(),
                "total_records": 1000,
                "no_default_pct": 80.0,
                "default_pct": 20.0,
            },
            "detailed_eda": {
                "shape": (1000, 9),
                "missing_report": pd.DataFrame({ 'column': [], 'missing_count': [], 'missing_pct': [] }),  # Добавлено для гарантии создания файла
            },
        }

        save_eda_results(results, temp_dir)

        # Проверяем, что файлы созданы
        assert (temp_dir / "numeric_representativeness.csv").exists()
        assert (temp_dir / "categorical_representativeness.csv").exists()  # Теперь создается даже если пустой
        assert (temp_dir / "eda_features.csv").exists()
        assert (temp_dir / "eda_target.csv").exists()
        assert (temp_dir / "eda_summary.json").exists()

        # Проверяем содержимое JSON
        with open(temp_dir / "eda_summary.json", 'r') as f:
            summary = json.load(f)
        assert "shape" in summary
        assert "missing_report" in summary

    def test_eda_functions_with_empty_dataframe(self):
        """Тест функций EDA с пустым DataFrame."""
        empty_df = pd.DataFrame()

        # find_target_column может вернуть последнюю колонку (пусто - ошибка)
        with pytest.raises(IndexError):
            find_target_column(empty_df)

        # analyze_representativeness должен обработать пустой DataFrame
        result = analyze_representativeness(empty_df, empty_df)
        assert isinstance(result, dict)
        assert "sampling_rate" in result["overall"]  # Теперь не падает

    def test_eda_functions_with_single_column(self):
        """Тест функций EDA с DataFrame с одной колонкой."""
        single_col_df = pd.DataFrame({"default.payment.next.month": [0, 1]})

        # find_target_column должен найти целевую переменную
        assert find_target_column(single_col_df) == "default.payment.next.month"

        # analyze_target_variable должен работать
        result = analyze_target_variable(single_col_df, "default.payment.next.month")
        assert isinstance(result, dict)
        assert "df_features" in result
        assert "y_target" in result
        assert result["df_features"].empty  # Поскольку одна колонка - таргет


class TestEDAPerformance:
    """Тесты производительности для EDA."""

    def test_large_dataset_performance(self):
        """Тест производительности на большом датасете."""
        # Создаем большой датасет
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "id": range(10000),
                "limit_bal": np.random.normal(150000, 50000, 10000),
                "age": np.random.normal(35, 10, 10000),
                "sex": np.random.choice(["Male", "Female"], 10000),
                "default.payment.next.month": np.random.choice([0, 1], 10000, p=[0.8, 0.2]),
            }
        )

        # Тест должен завершиться за разумное время
        import time

        start_time = time.time()

        # Анализируем репрезентативность сэмпла относительно полного датасета
        large_sample = large_data.sample(frac=0.1, random_state=42)
        result = analyze_representativeness(large_sample, large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # Проверяем, что выполнение заняло менее 10 секунд
        assert execution_time < 10
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])
