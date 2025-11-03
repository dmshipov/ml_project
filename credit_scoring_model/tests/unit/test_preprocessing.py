"""
Unit тесты для модуля предобработки (scripts/data_processing/preprocessing.py).
Тесты охватывают функции нагрузки данных, очистки, препроцессинга и сохранения.
Используется pytest с фикстурами для тестовых данных.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Добавляем путь к модулям (относительно tests/unit/)
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_processing.preprocessing import (
    apply_preprocessing,
    create_preprocessor,
    fillna_for_columns,
    find_and_remove_single_value_cols,
    find_high_correlation_cols,
    load_and_process_data,
    load_data,
)


@pytest.fixture
def sample_data():
    """Создает тестовые данные с различными типами колонок."""
    np.random.seed(42)
    data = {
        "id": range(100),
        "loan_amnt": np.random.normal(10000, 3000, 100),
        "int_rate": np.random.normal(12, 3, 100),
        "grade": np.random.choice(["A", "B", "C", "D"], 100),
        "emp_length": np.random.choice(["< 1 year", "1 year", "2 years"], 100),
        "annual_inc": np.random.lognormal(10, 0.5, 100),
        "target": np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        "single_val_col": ["const"] * 100,  # Один уникальный
        "high_missing_cat": ["a"] * 20 + [np.nan] * 80,  # После dropna() один уникальный
        "correl_col1": np.random.normal(5, 1, 100),  # Высокая корреляция с correl_col2
        "correl_col2": None,  # Создадим ниже
    }
    df = pd.DataFrame(data)
    # Высокая корреляция между correl_col1 и correl_col2
    df["correl_col2"] = df["correl_col1"] * 0.95 + np.random.normal(0, 0.1, 100)
    df = df.drop(columns=["target"])  # X без таргета
    return df


@pytest.fixture
def sample_target():
    """Создает тестовый таргет."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 100), name="default.payment.next.month")


@pytest.fixture
def temp_data_paths(sample_data, sample_target):
    """Создает временные файлы для X и y."""
    with tempfile.TemporaryDirectory() as temp_dir:
        x_path = Path(temp_dir) / "features.csv"
        y_path = Path(temp_dir) / "target.csv"
        sample_data.to_csv(x_path, index=False)
        sample_target.to_csv(y_path, index=False)
        yield x_path, y_path


class TestPreprocessing:
    """Основные тесты для функций предобработки."""

    def test_load_data(self, temp_data_paths, sample_data, sample_target):
        """Тест загрузки данных."""
        x_path, y_path = temp_data_paths
        X, y = load_data(str(x_path), str(y_path))

        # Проверяем типы и формы
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape[0] == sample_data.shape[0]
        assert y.shape[0] == sample_target.shape[0]
        # Колонки преобразованы в lowercase
        assert all(col.islower() for col in X.columns)
        # Имя таргета
        assert y.name == "default.payment.next.month"

    def test_find_and_remove_single_value_cols(self, sample_data):
        """Тест удаления колонок с одним уникальным значением."""
        df_clean, dropped_cols = find_and_remove_single_value_cols(sample_data)

        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(dropped_cols, list)
        assert "single_val_col" in dropped_cols
        assert "high_missing_cat" in dropped_cols  # После dropna() single value
        assert "single_val_col" not in df_clean.columns
        assert "high_missing_cat" not in df_clean.columns
        # Гибкий assert: удалено ровно столько, сколько в списке
        assert df_clean.shape[1] == sample_data.shape[1] - len(dropped_cols)

    def test_find_high_correlation_cols(self, sample_data):
        """Тест нахождения колонок с высокой корреляцией."""
        # sample_data имеет high_corr между correl_col1 и correl_col2 (>0.9)
        high_corr_cols = find_high_correlation_cols(sample_data, threshold=0.9)

        assert isinstance(high_corr_cols, list)
        assert len(high_corr_cols) >= 1  # Должна быть хотя бы одна
        # Проверяем, что одна из пар удалена (функция удаляет одну из коррелирующей пары)
        correlated_pairs = ["correl_col1", "correl_col2"]
        assert any(col in high_corr_cols for col in correlated_pairs)

    def test_fillna_for_columns(self, sample_data):
        """Тест заполнения пропусков в категориальных колонках."""
        df_with_nan = sample_data.copy()
        df_with_nan.loc[:, "high_missing_cat"] = np.nan  # Устанавливаем все в NaN, но в sample уже частично

        cat_cols = ["high_missing_cat", "grade"]
        df_filled = fillna_for_columns(df_with_nan, cat_cols)

        assert df_filled["high_missing_cat"].notna().all()
        # Мода ('a') должна заполнить большинство
        mode_val = df_filled["high_missing_cat"].mode().iloc[0] if not df_filled["high_missing_cat"].mode().empty else None
        assert mode_val is not None and len(df_filled[df_filled["high_missing_cat"] == mode_val]) > 0

    def test_load_and_process_data(self, temp_data_paths, sample_data, sample_target):
        """Тест полной загрузки и обработки данных."""
        x_path, y_path = temp_data_paths
        X_train, X_test, y_train, y_test, columns_to_process, cat_cols = load_and_process_data(
            str(x_path), str(y_path), test_size=0.2, random_state=42
        )

        # Проверяем типы
        assert all(isinstance(item, (pd.DataFrame, pd.Series, list)) for item in [X_train, X_test, y_train, y_test, columns_to_process, cat_cols])

        # Размеры после обработки
        total_samples = sample_data.shape[0]
        assert len(X_train) == int(total_samples * 0.8)
        assert len(X_test) == total_samples - len(X_train)

        # Категориальные колонки
        assert all(col in X_train.columns for col in cat_cols)
        assert len(cat_cols) > 0  # Должны быть категориальные

        # Нет колонок с одним значением и высокой корреляцией
        remaining_cols = X_train.columns
        assert "single_val_col" not in remaining_cols
        assert "high_missing_cat" not in remaining_cols  # Удалена как single_value

    def test_create_preprocessor(self, sample_data):
        """Тест создания препроцессора."""
        numeric_cols = ["loan_amnt", "int_rate", "annual_inc"]
        cat_cols = ["grade", "emp_length"]

        preprocessor = create_preprocessor(numeric_cols, cat_cols)

        assert isinstance(preprocessor, ColumnTransformer)
        assert len(preprocessor.transformers) == 2  # num и cat
        # Fit перед проверкой named_transformers_
        fit_data = sample_data[numeric_cols + cat_cols].copy()
        fit_data = fit_data.fillna(0)  # Чтобы fit не сломался на NaN
        preprocessor.fit(fit_data)
        # Проверяем компоненты
        num_transformer = preprocessor.named_transformers_["num"]
        cat_transformer = preprocessor.named_transformers_["cat"]
        assert isinstance(num_transformer, StandardScaler)
        assert isinstance(cat_transformer, OneHotEncoder)

    def test_apply_preprocessing(self, temp_data_paths, sample_data, sample_target):
        """Тест применения препроцессинга и сохранения."""
        x_path, y_path = temp_data_paths
        # Сначала получим processed данные через load_and_process_data
        X_train, X_test, y_train, y_test, columns_to_process, cat_cols = load_and_process_data(
            str(x_path), str(y_path), test_size=0.2, random_state=42
        )

        # Создаем временный dir для сохранения
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "processed"
            X_train_proc, X_test_proc, preprocessor = apply_preprocessing(
                X_train, X_test, y_train, y_test, columns_to_process, cat_cols, str(output_dir)
            )

            # Проверяем типы
            assert isinstance(X_train_proc, pd.DataFrame)
            assert isinstance(X_test_proc, pd.DataFrame)
            assert isinstance(preprocessor, ColumnTransformer)

            # Размеры processed (упрощённо: после onehot > numeric cols)
            num_cols = [col for col in columns_to_process if col not in cat_cols]
            assert X_train_proc.shape[1] > len(num_cols)  # Onehot добавляет колонки

            # Файлы сохранены
            assert (output_dir / "X_train.csv").exists()
            assert (output_dir / "preprocessor.pkl").exists()
            assert (output_dir / "X_train_processed.csv").exists()


class TestPreprocessingEdgeCases:
    """Тесты для граничных случаев."""

    def test_load_data_no_columns(self):
        """Тест загрузки данных без колонок (но с заголовками)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            x_path = Path(temp_dir) / "empty_x.csv"
            y_path = Path(temp_dir) / "empty_y.csv"
            # Создаём с колонками, но 0 строк
            pd.DataFrame(columns=["id", "loan_amnt"]).to_csv(x_path, index=False)
            pd.DataFrame(columns=["default.payment.next.month"]).to_csv(y_path, index=False)
            X, y = load_data(str(x_path), str(y_path))
            assert X.empty and X.shape[1] == 2  # Колонки есть, строк 0
            assert y.empty and y.shape[0] == 0

    def test_find_and_remove_single_value_cols_no_single(self):
        """Нет колонок с одним значением."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df_clean, dropped = find_and_remove_single_value_cols(df)
        assert dropped == []
        pd.testing.assert_frame_equal(df_clean, df)

    def test_find_high_correlation_cols_no_corr(self):
        """Нет колонок с высокой корреляцией."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [10, 20, 15]})  # corr ~0.5
        dropped = find_high_correlation_cols(df, threshold=0.9)
        assert dropped == []

    def test_fillna_for_columns_no_cols(self):
        """Нет колонок для заполнения."""
        df = pd.DataFrame({"num": [1, 2]})
        df_filled = fillna_for_columns(df, [])
        pd.testing.assert_frame_equal(df_filled, df)

    def test_load_and_process_data_stratify_fail(self, sample_data, sample_target):
        """Stratify падает, fallback к random split."""
        # Создаем таргет с одними значениями для stratify fail
        bad_target = pd.Series([0] * 100, name="default.payment.next.month")
        with tempfile.TemporaryDirectory() as temp_dir:
            x_path = Path(temp_dir) / "features.csv"
            y_path = Path(temp_dir) / "target.csv"
            sample_data.to_csv(x_path, index=False)
            bad_target.to_csv(y_path, index=False)

            with patch("warnings.warn") as mock_warn:
                X_train, X_test, y_train, y_test, _, _ = load_and_process_data(
                    str(x_path), str(y_path), test_size=0.2
                )
                mock_warn.assert_called_once()  # Предупреждение о fallback
                assert len(X_train) == 80
                assert len(X_test) == 20

    def test_create_preprocessor_empty_lists(self):
        """Пустые списки колонок."""
        preprocessor = create_preprocessor([], [])
        assert isinstance(preprocessor, ColumnTransformer)
        # Должен работать без ошибок (remainder='passthrough' по умолчанию?)

    def test_apply_preprocessing_empty_data(self, sample_data):
        """Пустые данные."""
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=int)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "processed"
            X_train_proc, X_test_proc, preprocessor = apply_preprocessing(
                empty_df, empty_df, empty_series, empty_series, [], [], str(output_dir)
            )
            assert X_train_proc.empty
            assert X_test_proc.empty
            assert isinstance(preprocessor, ColumnTransformer)


if __name__ == "__main__":
    pytest.main([__file__])
