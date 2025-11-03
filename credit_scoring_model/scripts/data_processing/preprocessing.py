"""
Скрипт для предобработки данных кредитного скоринга.

Этот скрипт выполняет:
1. Очистку данных от пропусков и выбросов
2. Удаление ненужных столбцов
3. Обработку категориальных и числовых признаков
4. Разделение на обучающую и тестовую выборки
5. Сохранение обработанных данных

"""

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

warnings.filterwarnings("ignore")


def load_data(features_path: str, target_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads data: X from features_path and y from target_path.
    """
    print("Loading data...")
    X = pd.read_csv(features_path)
    X.columns = X.columns.str.lower()
    
    y_df = pd.read_csv(target_path)
    if y_df.shape[1] == 1:
        y = y_df.iloc[:, 0]
    else:
        y = y_df.squeeze()  # If one column but multiple rows
    y = pd.Series(y, name="default.payment.next.month")  # Set name as in original
    
    print(f"Loaded X: {X.shape[0]} records, {X.shape[1]} features.")
    print(f"Loaded y: {y.shape[0]} records.")
    print(f"X columns: {list(X.columns)}")
    print(f"y column: {y.name}")
    return X, y


def find_and_remove_single_value_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Finds and removes columns with a single unique value.
    """
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_drop.append(col)
            print(f"Column with single value: {col}")
    if cols_to_drop:
        df = df.drop(cols_to_drop, axis=1)
    print(f"Removed columns with single value: {len(cols_to_drop)}")
    return df, cols_to_drop


def find_high_correlation_cols(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """
    Finds columns with high correlation on df (already without target).
    """
    # df here is already X (without target), so no need to check target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    else:
        to_drop = []
    print(f"Columns with high correlation (> {threshold}): {to_drop}")
    return to_drop


def fillna_for_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Handling missing values for categorical columns.
    """
    for col in cols:
        if col in df.columns:  # Added to avoid KeyError
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'MISSING'
            df[col] = df[col].fillna(mode_val)
    return df


def load_and_process_data(
    features_path: str,
    target_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Loads X and y separately, removes unnecessary columns, handles missing values, splits into train/test.
    """
    # Load X and y separately
    data_X, data_y = load_data(features_path, target_path)

    # Removal of columns with single value from X
    data_X, cols_single = find_and_remove_single_value_cols(data_X)

    # Find columns with high correlation (target already excluded - it's in y)
    cols_high_corr = find_high_correlation_cols(data_X)
    print(f"High correlation columns: {cols_high_corr}")

    # Columns to drop (only from X)
    cols_to_drop = list(set(cols_single + cols_high_corr))  # Remove duplicates
    print(f"Total unique columns to drop in X: {cols_to_drop}")

    # Removal of unnecessary columns from X (check for existence)
    existing_cols_to_drop = [col for col in cols_to_drop if col in data_X.columns]
    if existing_cols_to_drop:
        X = data_X.drop(existing_cols_to_drop, axis=1)
    else:
        X = data_X
    y = data_y  # y remains as is

    # Now columns_to_process only from X (after all drops)
    columns_to_process = X.columns.tolist()

    print(f"X shape after drop: {X.shape}, y shape: {y.shape}")
    print(f"Columns to process: {len(columns_to_process)}")

    # Splitting into train/test with fallback for stratify, including check for single unique class
    if y.nunique() == 1:
        warnings.warn("Only one unique class in target. Falling back to random split without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            warnings.warn(f"Stratified split failed: {e}. Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    print(f"Split: X_train {X_train.shape}, X_test {X_test.shape}")

    # Categorical columns (only from X.columns)
    cat_cols = [col for col in X.columns if X[col].dtype == 'object']

    print(f"Categorical columns: {cat_cols}")

    # Handling missing values on train (and copy for safety)
    X_train = fillna_for_columns(X_train.copy(), cat_cols)
    print("Missing values handled on train.")
    print(f"Final: X_train {X_train.shape}, y_train {y_train.shape}")
    return X_train, X_test, y_train, y_test, columns_to_process, cat_cols


def create_preprocessor(
    numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    """
    Creates a preprocessor for feature processing.
    """
    print("\nCreating preprocessor...")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )

    print(
        f"Preprocessor created for {len(numeric_columns)} numeric and {len(categorical_columns)} categorical features"
    )
    return preprocessor


def apply_preprocessing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    columns_to_process: List[str],
    cat_cols: List[str],
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Applies the preprocessor to the data, saves processed data in CSV for diagnostics.
    """
    from sklearn import set_config
    set_config(transform_output="pandas")  # Make output DataFrame with names

    print("Fitting preprocessor on train...")

    # Define numeric and categorical columns
    numeric_columns = [col for col in columns_to_process if col not in cat_cols]
    categorical_columns = cat_cols

    preprocessor = create_preprocessor(numeric_columns, categorical_columns)

    # Fitting preprocessor (fit on train)
    X_train_processed = preprocessor.fit_transform(X_train)  # Now DataFrame
    X_test_processed = preprocessor.transform(X_test)  # DataFrame

    print(f"Processed: X_train {X_train_processed.shape}, X_test {X_test_processed.shape}")

    # Saving raw X_train and X_test (before preprocessing)
    Path(output_dir).mkdir(exist_ok=True)
    X_train.to_csv(Path(output_dir) / "X_train.csv", index=False)
    X_test.to_csv(Path(output_dir) / "X_test.csv", index=False)
    y_train.to_csv(Path(output_dir) / "y_train.csv", index=False)
    y_test.to_csv(Path(output_dir) / "y_test.csv", index=False)

    # Save preprocessor for use in training.py
    joblib.dump(preprocessor, Path(output_dir) / "preprocessor.pkl")
    print(f"Preprocessor saved: {Path(output_dir) / 'preprocessor.pkl'}")

    # Optionally save processed data for debugging
    X_train_processed.to_csv(Path(output_dir) / "X_train_processed.csv", index=False)
    X_test_processed.to_csv(Path(output_dir) / "X_test_processed.csv", index=False)

    print(f"Saved to {output_dir}")
    return X_train_processed, X_test_processed, preprocessor


def main():
    """
    Main function for data processing.

    """
    features_path = "data/processed/eda_features.csv"
    target_path = "data/processed/eda_target.csv"
    
    X_train, X_test, y_train, y_test, columns_to_process, cat_cols = load_and_process_data(
        features_path, target_path, test_size=0.2, random_state=42
    )

    # Apply preprocessing and saving
    apply_preprocessing(
        X_train, X_test, y_train, y_test, columns_to_process, cat_cols
    )

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()