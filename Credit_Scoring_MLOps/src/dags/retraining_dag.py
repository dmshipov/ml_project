from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import (
    KubernetesPodOperator,
)


def dummy_validate_model(**context) -> None:
    print("Валидация новой модели выполнена условно успешно.")


default_args = {
    "owner": "mlops_student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="credit_scoring_retraining",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 3 * * *",
    catchup=False,
    tags=["credit", "retraining"],
) as dag:
    check_drift = KubernetesPodOperator(
        task_id="check_data_drift",
        name="check-data-drift",
        namespace="default",
        image="REGISTRY/credit-scoring-tools:latest",
        cmds=["python", "monitoring/evidently_drift.py"],
        is_delete_operator_pod=True,
    )

    retrain_model = KubernetesPodOperator(
        task_id="retrain_model",
        name="retrain-model",
        namespace="default",
        image="REGISTRY/credit-scoring-trainer:latest",
        cmds=[
            "python",
            "src/scripts/train_model.py",
            "--data",
            "data/train.csv",
        ],
        is_delete_operator_pod=True,
    )

    validate_model = PythonOperator(
        task_id="validate_new_model",
        python_callable=dummy_validate_model,
    )

    deploy_canary = KubernetesPodOperator(
        task_id="deploy_canary_release",
        name="deploy-canary",
        namespace="default",
        image="bitnami/kubectl:latest",
        cmds=[
            "kubectl",
            "apply",
            "-f",
            "kubernetes/deployment-api.yaml",
        ],
        is_delete_operator_pod=True,
    )

    check_drift >> retrain_model >> validate_model >> deploy_canary
