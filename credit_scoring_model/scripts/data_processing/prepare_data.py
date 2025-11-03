#!/usr/bin/env python3
"""
Data preparation script for credit scoring system.
This script processes raw data and prepares it for model training.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processor for credit scoring data."""

    def __init__(self, data_path: str, output_path: str):
        """Initialize data processor."""
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.scalers = {}
        self.encoders = {}

        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "processed").mkdir(exist_ok=True)
        (self.output_path / "artifacts").mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load raw data."""
        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.suffix == ".csv":
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix == ".parquet":
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset."""
        logger.info("Cleaning data...")

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Handle missing values
        missing_before = df.isnull().sum().sum()

        # Fill missing values based on column type
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("unknown")

        missing_after = df.isnull().sum().sum()
        logger.info(f"Filled {missing_before - missing_after} missing values")

        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable from loan status."""
        logger.info("Creating target variable...")

        # Map loan status to binary target
        def map_loan_status(status):
            if pd.isna(status):
                return 1  # Default to bad loan for missing values

            status_str = str(status).strip().lower()

            # Good loans
            if any(
                good in status_str for good in ["fully paid", "current", "good loan"]
            ):
                return 0
            # Bad loans
            else:
                return 1

        df["target"] = df["loan_status"].apply(map_loan_status)

        # Log target distribution
        target_dist = df["target"].value_counts()
        logger.info(f"Target distribution: {target_dist.to_dict()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        logger.info("Engineering features...")

        # FICO score average
        if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
            df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

        # Loan to income ratio
        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income_ratio"] = df["loan_amnt"] / (
                df["annual_inc"] + 1
            )  # Add 1 to avoid division by zero

        # Employment length to numeric
        if "emp_length" in df.columns:
            df["emp_length_numeric"] = df["emp_length"].apply(
                self._convert_emp_length_to_numeric
            )

        # Debt to income ratio categories
        if "dti" in df.columns:
            df["dti_category"] = pd.cut(
                df["dti"],
                bins=[0, 10, 20, 30, 100],
                labels=["low", "medium", "high", "very_high"],
            )

        return df

    def _convert_emp_length_to_numeric(self, emp_length: str) -> int:
        """Convert employment length to numeric value."""
        if pd.isna(emp_length) or emp_length == "n/a":
            return 0

        if "< 1 year" in str(emp_length):
            return 0
        elif "10+ years" in str(emp_length):
            return 10
        else:
            try:
                return int(str(emp_length).split()[0])
            except (ValueError, IndexError):
                return 0

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Select features for model training."""
        logger.info("Selecting features...")

        # Define feature columns
        feature_columns = [
            "annual_inc",
            "loan_amnt",
            "fico_range_low",
            "fico_range_high",
            "dti",
            "revol_util",
            "inq_last_6mths",
            "delinq_2yrs",
            "pub_rec",
            "fico_avg",
            "loan_to_income_ratio",
            "emp_length_numeric",
        ]

        # Select only available features
        available_features = [col for col in feature_columns if col in df.columns]
        logger.info(
            f"Selected {len(available_features)} features: {available_features}"
        )

        X = df[available_features].copy()
        y = df["target"].copy()

        return X, y

    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess features."""
        logger.info("Preprocessing features...")

        X_processed = X.copy()

        # Handle categorical variables
        categorical_columns = X_processed.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_columns:
            if fit:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.encoders[col] = le
            else:
                if col in self.encoders:
                    X_processed[col] = self.encoders[col].transform(
                        X_processed[col].astype(str)
                    )
                else:
                    X_processed[col] = 0

        # Scale numerical features
        numerical_columns = X_processed.select_dtypes(include=[np.number]).columns

        for col in numerical_columns:
            if fit:
                scaler = StandardScaler()
                X_processed[col] = scaler.fit_transform(X_processed[[col]])
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    X_processed[col] = self.scalers[col].transform(X_processed[[col]])

        return X_processed

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        logger.info("Splitting data into train and test sets...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """Save processed data."""
        logger.info("Saving processed data...")

        # Save data
        X_train.to_csv(self.output_path / "processed" / "X_train.csv", index=False)
        X_test.to_csv(self.output_path / "processed" / "X_test.csv", index=False)
        y_train.to_csv(self.output_path / "processed" / "y_train.csv", index=False)
        y_test.to_csv(self.output_path / "processed" / "y_test.csv", index=False)

        # Save preprocessors
        joblib.dump(self.scalers, self.output_path / "artifacts" / "scalers.pkl")
        joblib.dump(self.encoders, self.output_path / "artifacts" / "encoders.pkl")

        logger.info("Processed data saved successfully")

    def process(self):
        """Main processing pipeline."""
        logger.info("Starting data processing pipeline...")

        # Load data
        df = self.load_data()

        # Clean data
        df = self.clean_data(df)

        # Create target variable
        df = self.create_target_variable(df)

        # Engineer features
        df = self.engineer_features(df)

        # Select features
        X, y = self.select_features(df)

        # Preprocess features
        X_processed = self.preprocess_features(X, fit=True)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X_processed, y)

        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test)

        logger.info("Data processing pipeline completed successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare data for credit scoring model"
    )
    parser.add_argument("--data-path", required=True, help="Path to raw data file")
    parser.add_argument(
        "--output-path", required=True, help="Path to save processed data"
    )

    args = parser.parse_args()

    # Create data processor
    processor = DataProcessor(args.data_path, args.output_path)

    # Process data
    processor.process()


if __name__ == "__main__":
    main()
