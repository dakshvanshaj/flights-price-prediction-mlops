from airflow.models.dag import DAG
import pendulum
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
    dag_id="data_pipeline_v1",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    description="This dag runs individual Bronze, Silver and Gold pipelines.",
    schedule=None,
    catchup=False,
    tags=["data_pipeline", "pipeline", "bronze", "silver", "gold"],
    default_args=default_args,
) as dag:
    # Task 1: Bronze Layer
    bronze_pipeline = PythonOperator(
        task_id="run_bronze_pipeline_and_validate", python_callable=run_bronze_pipeline
    )

    # Task 2: Silver Layer
    silver_pipeline = PythonOperator(
        task_id="run_silver_pipeline_and_validate", python_callable=run_silver_pipeline
    )

    # Task 3: Gold Layer
    gold_pipeline = PythonOperator(
        task_id="run_gold_pipeline_and_validate", python_callable=run_gold_pipeline
    )

    # Set dependencies using the task objects  created
    bronze_pipeline >> silver_pipeline >> gold_pipeline
