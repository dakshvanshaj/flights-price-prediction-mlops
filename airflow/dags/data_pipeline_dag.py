from airflow.models.dag import DAG
import pendulum
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "Daksh",
    "retries": 0,
    "retry_delay": pendulum.duration(minutes=5),
}

with DAG(
    dag_id="dvc_repro_pipeline",
    start_date=pendulum.datetime(2024, 8, 1, tz="UTC"),
    description="This DAG runs the main DVC pipeline using 'dvc repro'.",
    schedule=None,
    catchup=False,
    tags=["flights_mlops", "dvc", "bash"],
    default_args=default_args,
) as dag:
    # This task now correctly changes directory to the project root
    # inside the container before running the dvc command.
    run_dvc_pipeline = BashOperator(
        task_id="run_dvc_repro",
        bash_command="cd /opt/airflow/Flights_Mlops && dvc repro",
    )
