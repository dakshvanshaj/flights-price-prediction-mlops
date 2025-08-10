from airflow.models.dag import DAG
import pendulum
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Import your pipeline functions
from pipelines.bronze_pipeline import main as run_bronze_pipeline
from pipelines.gold_pipeline import main as run_gold_pipeline
from pipelines.silver_pipeline import main as run_silver_pipeline

default_args = {
    "owner": "Daksh",
    "retries": 0,
    "retry_delay": pendulum.duration(minutes=5),
}

with DAG(
    dag_id="data_pipeline",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    description="Fetches data with DVC and runs the Bronze, Silver, and Gold pipelines.",
    schedule=None,
    catchup=False,
    tags=["production", "dvc", "flights_mlops"],
    default_args=default_args,
) as dag:
    # Task 1: Fetch data using DVC. This task must succeed before any others run.
    fetch_data = BashOperator(
        task_id="fetch_data_with_dvc",
        bash_command="cd /opt/airflow/Flights_Mlops && dvc pull",
    )

    # Task 2: Bronze Layer
    bronze_pipeline_task = PythonOperator(
        task_id="run_bronze_pipeline", python_callable=run_bronze_pipeline
    )

    # Task 3: Silver Layer
    silver_pipeline_task = PythonOperator(
        task_id="run_silver_pipeline", python_callable=run_silver_pipeline
    )

    # Task 4: Gold Layer
    gold_pipeline_task = PythonOperator(
        task_id="run_gold_pipeline", python_callable=run_gold_pipeline
    )

    # Now, the pipeline will only start after the data is successfully pulled.
    fetch_data >> bronze_pipeline_task >> silver_pipeline_task >> gold_pipeline_task
