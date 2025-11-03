"""
Общие фикстуры для тестов.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings.config import Settings


@pytest.fixture
def sample_credit_data():
    """Создает тестовые данные кредитного скоринга."""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "id": range(n_samples),
        "loan_amnt": np.random.normal(10000, 3000, n_samples),
        "int_rate": np.random.normal(12, 3, n_samples),
        "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
        "sub_grade": np.random.choice(["A1", "A2", "A3", "A4", "A5"], n_samples),
        "emp_length": np.random.choice(
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years"],
            n_samples,
        ),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE"], n_samples),
        "annual_inc": np.random.lognormal(10, 0.5, n_samples),
        "verification_status": np.random.choice(
            ["Verified", "Not Verified", "Source Verified"], n_samples
        ),
        "loan_status": np.random.choice(
            ["Fully Paid", "Charged Off", "Current"], n_samples, p=[0.7, 0.2, 0.1]
        ),
        "purpose": np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "other"],
            n_samples,
        ),
        "addr_state": np.random.choice(["CA", "NY", "TX", "FL", "IL"], n_samples),
        "dti": np.random.normal(15, 5, n_samples),
        "delinq_2yrs": np.random.poisson(0.5, n_samples),
        "earliest_cr_line": pd.date_range(
            "1990-01-01", "2020-01-01", periods=n_samples
        ),
        "inq_last_6mths": np.random.poisson(1, n_samples),
        "mths_since_last_delinq": np.random.exponential(12, n_samples),
        "mths_since_last_record": np.random.exponential(24, n_samples),
        "open_acc": np.random.poisson(8, n_samples),
        "pub_rec": np.random.poisson(0.2, n_samples),
        "revol_bal": np.random.lognormal(8, 1, n_samples),
        "revol_util": np.random.normal(50, 20, n_samples),
        "total_acc": np.random.poisson(15, n_samples),
        "initial_list_status": np.random.choice(["f", "w"], n_samples),
        "out_prncp": np.random.lognormal(6, 1, n_samples),
        "out_prncp_inv": np.random.lognormal(6, 1, n_samples),
        "total_pymnt": np.random.lognormal(8, 1, n_samples),
        "total_pymnt_inv": np.random.lognormal(8, 1, n_samples),
        "total_rec_prncp": np.random.lognormal(7, 1, n_samples),
        "total_rec_int": np.random.lognormal(6, 1, n_samples),
        "total_rec_late_fee": np.random.exponential(10, n_samples),
        "recoveries": np.random.exponential(50, n_samples),
        "collection_recovery_fee": np.random.exponential(20, n_samples),
        "last_pymnt_d": pd.date_range("2015-01-01", "2020-01-01", periods=n_samples),
        "last_pymnt_amnt": np.random.lognormal(6, 1, n_samples),
        "next_pymnt_d": pd.date_range("2020-01-01", "2025-01-01", periods=n_samples),
        "last_credit_pull_d": pd.date_range(
            "2015-01-01", "2020-01-01", periods=n_samples
        ),
        "last_fico_range_high": np.random.normal(700, 50, n_samples),
        "last_fico_range_low": np.random.normal(650, 50, n_samples),
        "collections_12_mths_ex_med": np.random.poisson(0.1, n_samples),
        "mths_since_last_major_derog": np.random.exponential(36, n_samples),
        "policy_code": np.ones(n_samples),
        "application_type": ["Individual"] * n_samples,
        "annual_inc_joint": np.random.lognormal(10, 0.5, n_samples),
        "dti_joint": np.random.normal(15, 5, n_samples),
        "verification_status_joint": np.random.choice(
            ["Verified", "Not Verified"], n_samples
        ),
        "acc_now_delinq": np.random.poisson(0.1, n_samples),
        "tot_coll_amt": np.random.exponential(100, n_samples),
        "tot_cur_bal": np.random.lognormal(9, 1, n_samples),
        "open_acc_6m": np.random.poisson(2, n_samples),
        "open_act_il": np.random.poisson(3, n_samples),
        "open_il_12m": np.random.poisson(1, n_samples),
        "open_il_24m": np.random.poisson(2, n_samples),
        "mths_since_rcnt_il": np.random.exponential(12, n_samples),
        "total_bal_il": np.random.lognormal(8, 1, n_samples),
        "il_util": np.random.normal(30, 15, n_samples),
        "open_rv_12m": np.random.poisson(1, n_samples),
        "open_rv_24m": np.random.poisson(2, n_samples),
        "max_bal_bc": np.random.lognormal(8, 1, n_samples),
        "all_util": np.random.normal(40, 20, n_samples),
        "total_rev_hi_lim": np.random.lognormal(9, 1, n_samples),
        "inq_fi": np.random.poisson(0.5, n_samples),
        "total_cu_tl": np.random.poisson(5, n_samples),
        "inq_last_12m": np.random.poisson(2, n_samples),
        "acc_open_past_24mths": np.random.poisson(3, n_samples),
        "avg_cur_bal": np.random.lognormal(8, 1, n_samples),
        "bc_open_to_buy": np.random.lognormal(6, 1, n_samples),
        "bc_util": np.random.normal(35, 20, n_samples),
        "chargeoff_within_12_mths": np.random.poisson(0.1, n_samples),
        "delinq_amnt": np.random.exponential(100, n_samples),
        "mo_sin_old_il_acct": np.random.exponential(60, n_samples),
        "mo_sin_old_rev_tl_op": np.random.exponential(48, n_samples),
        "mo_sin_rcnt_rev_tl_op": np.random.exponential(12, n_samples),
        "mo_sin_rcnt_tl": np.random.exponential(6, n_samples),
        "mort_acc": np.random.poisson(2, n_samples),
        "mths_since_recent_bc": np.random.exponential(6, n_samples),
        "mths_since_recent_bc_dlq": np.random.exponential(12, n_samples),
        "mths_since_recent_inq": np.random.exponential(3, n_samples),
        "mths_since_recent_revol_delinq": np.random.exponential(18, n_samples),
        "num_accts_ever_120_pd": np.random.poisson(0.5, n_samples),
        "num_actv_bc_tl": np.random.poisson(3, n_samples),
        "num_actv_rev_tl": np.random.poisson(2, n_samples),
        "num_bc_sats": np.random.poisson(5, n_samples),
        "num_bc_tl": np.random.poisson(8, n_samples),
        "num_il_tl": np.random.poisson(2, n_samples),
        "num_op_rev_tl": np.random.poisson(4, n_samples),
        "num_rev_accts": np.random.poisson(6, n_samples),
        "num_rev_tl_bal_gt_0": np.random.poisson(4, n_samples),
        "num_sats": np.random.poisson(8, n_samples),
        "num_tl_120dpd_2m": np.random.poisson(0.1, n_samples),
        "num_tl_30dpd": np.random.poisson(0.2, n_samples),
        "num_tl_90g_dpd_24m": np.random.poisson(0.1, n_samples),
        "num_tl_op_past_12m": np.random.poisson(2, n_samples),
        "pct_tl_nvr_dlq": np.random.normal(95, 10, n_samples),
        "percent_bc_gt_75": np.random.normal(20, 15, n_samples),
        "pub_rec_bankruptcies": np.random.poisson(0.1, n_samples),
        "tax_liens": np.random.poisson(0.05, n_samples),
        "tot_hi_cred_lim": np.random.lognormal(9, 1, n_samples),
        "total_bal_ex_mort": np.random.lognormal(8, 1, n_samples),
        "total_bc_limit": np.random.lognormal(8, 1, n_samples),
        "total_il_high_credit_limit": np.random.lognormal(7, 1, n_samples),
        "revol_bal_joint": np.random.lognormal(8, 1, n_samples),
        "sec_app_fico_range_low": np.random.normal(650, 50, n_samples),
        "sec_app_fico_range_high": np.random.normal(700, 50, n_samples),
        "sec_app_earliest_cr_line": pd.date_range(
            "1990-01-01", "2020-01-01", periods=n_samples
        ),
        "sec_app_inq_last_6mths": np.random.poisson(1, n_samples),
        "sec_app_mort_acc": np.random.poisson(2, n_samples),
        "sec_app_open_acc": np.random.poisson(5, n_samples),
        "sec_app_revol_util": np.random.normal(30, 15, n_samples),
        "sec_app_open_act_il": np.random.poisson(1, n_samples),
        "sec_app_num_rev_accts": np.random.poisson(3, n_samples),
        "sec_app_chargeoff_within_12_mths": np.random.poisson(0.1, n_samples),
        "sec_app_collections_12_mths_ex_med": np.random.poisson(0.1, n_samples),
        "sec_app_mths_since_last_major_derog": np.random.exponential(36, n_samples),
        "hardship_flag": np.random.choice(["N", "Y"], n_samples, p=[0.95, 0.05]),
        "hardship_type": np.random.choice(
            ["Other", "Medical", "DebtConsolidation"], n_samples
        ),
        "hardship_reason": np.random.choice(
            ["Other", "Medical", "DebtConsolidation"], n_samples
        ),
        "hardship_status": np.random.choice(
            ["Approved", "Denied", "In Progress"], n_samples
        ),
        "deferral_term": np.random.exponential(3, n_samples),
        "hardship_amount": np.random.lognormal(7, 1, n_samples),
        "hardship_start_date": pd.date_range(
            "2015-01-01", "2020-01-01", periods=n_samples
        ),
        "hardship_end_date": pd.date_range(
            "2015-01-01", "2020-01-01", periods=n_samples
        ),
        "payment_plan_start_date": pd.date_range(
            "2015-01-01", "2020-01-01", periods=n_samples
        ),
        "hardship_length": np.random.exponential(6, n_samples),
        "hardship_dpd": np.random.poisson(2, n_samples),
        "hardship_loan_status": np.random.choice(["Current", "Paid Off"], n_samples),
        "orig_projected_additional_accrued_interest": np.random.exponential(
            100, n_samples
        ),
        "hardship_payoff_balance_amount": np.random.lognormal(7, 1, n_samples),
        "hardship_last_payment_amount": np.random.lognormal(6, 1, n_samples),
        "debt_settlement_flag": np.random.choice(["N", "Y"], n_samples, p=[0.99, 0.01]),
        "debt_settlement_flag_date": pd.date_range(
            "2015-01-01", "2020-01-01", periods=n_samples
        ),
        "settlement_status": np.random.choice(["Settled", "In Progress"], n_samples),
        "settlement_date": pd.date_range("2015-01-01", "2020-01-01", periods=n_samples),
        "settlement_amount": np.random.lognormal(7, 1, n_samples),
        "settlement_percentage": np.random.normal(50, 20, n_samples),
        "settlement_term": np.random.exponential(12, n_samples),
        "target": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    }

    df = pd.DataFrame(data)

    # Добавляем некоторые пропуски для реалистичности
    df.loc[np.random.choice(df.index, 50), "mths_since_last_delinq"] = np.nan
    df.loc[np.random.choice(df.index, 30), "mths_since_last_record"] = np.nan
    df.loc[np.random.choice(df.index, 20), "mths_since_last_major_derog"] = np.nan

    return df


@pytest.fixture
def sample_processed_data():
    """Создает обработанные тестовые данные."""
    np.random.seed(42)
    n_samples = 500

    data = {
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.normal(0, 1, n_samples),
        "feature3": np.random.normal(0, 1, n_samples),
        "feature4": np.random.normal(0, 1, n_samples),
        "target": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_model():
    """Создает тестовую модель."""
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    return model


@pytest.fixture
def temp_directory():
    """Создает временную директорию."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_csv_file(sample_credit_data):
    """Создает временный CSV файл с данными."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_credit_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_model_file(sample_model, sample_processed_data):
    """Создает временный файл модели."""
    # Обучаем модель на тестовых данных
    X = sample_processed_data.drop(columns=["target"])
    y = sample_processed_data["target"]
    sample_model.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        joblib.dump(sample_model, f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_mlflow():
    """Мокает MLflow для тестов."""
    with patch("mlflow.set_tracking_uri"), patch("mlflow.create_experiment"), patch(
        "mlflow.set_experiment"
    ), patch("mlflow.start_run") as mock_start_run, patch("mlflow.log_params"), patch(
        "mlflow.log_metrics"
    ), patch(
        "mlflow.log_model"
    ):
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        yield mock_start_run


@pytest.fixture
def sample_config():
    """Создает тестовую конфигурацию."""
    return {
        "missing_threshold": 0.1,
        "outlier_threshold": 3.0,
        "correlation_threshold": 0.95,
        "duplicate_threshold": 0.05,
        "min_unique_ratio": 0.1,
        "max_skewness": 2.0,
        "max_kurtosis": 3.0,
        "drift_threshold": 0.1,
        "performance_threshold": 0.05,
        "bias_threshold": 0.1,
        "min_samples": 100,
    }


@pytest.fixture(scope="session")
def test_data_directory():
    """Создает директорию с тестовыми данными для сессии."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "data" / "raw"
        data_dir.mkdir(parents=True)

        # Создаем тестовый CSV файл
        test_data = pd.DataFrame(
            {
                "id": range(100),
                "loan_amnt": np.random.normal(10000, 3000, 100),
                "int_rate": np.random.normal(12, 3, 100),
                "grade": np.random.choice(["A", "B", "C", "D"], 100),
                "loan_status": np.random.choice(
                    ["Fully Paid", "Charged Off"], 100, p=[0.8, 0.2]
                ),
            }
        )

        test_data.to_csv(data_dir / "accepted_2007_to_2018Q4.csv", index=False)

        yield temp_dir


@pytest.fixture
def sample_metrics():
    """Создает тестовые метрики."""
    return {
        "accuracy": 0.95,
        "precision": 0.90,
        "recall": 0.85,
        "f1_score": 0.87,
        "roc_auc": 0.92,
        "average_precision": 0.88,
    }


@pytest.fixture
def sample_model_params():
    """Создает тестовые параметры модели."""
    return {
        "C": 1.0,
        "random_state": 42,
        "max_iter": 1000,
        "penalty": "l2",
        "solver": "liblinear",
    }


@pytest.fixture
def test_settings():
    """Создает тестовые настройки для приложения."""
    return Settings(
        database_url="sqlite:///test.db",
        secret_key="test-secret-key",
        jwt_secret_key="test-jwt-secret-key",
    )
