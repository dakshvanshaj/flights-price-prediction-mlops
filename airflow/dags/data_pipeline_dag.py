from airflow.models.dag import DAG
import pendulum
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


default_args = {"owner": "Daksh", "retries": 0}

with DAG(
    dag_id="data_preprocessing_pipeline",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    description="Fetches DVC data and runs the full ML pipeline.",
    schedule=None,
    catchup=False,
    tags=["data", "preprocessing", "pipeline", "dvc"],
    default_args=default_args,
) as dag:
    fetch_data = BashOperator(
        task_id="fetch_data_with_dvc",
        bash_command="cd /opt/airflow/Flights_Mlops && echo --- CWD --- && pwd && echo --- ls -la --- && ls -la && echo --- dvc pull --- && dvc pull -v",
    )

    bronze_pipeline_task = BashOperator(
        task_id="run_bronze_pipeline",
        bash_command=(
            "cd /opt/airflow/Flights_Mlops && "
            "python src/pipelines/bronze_pipeline.py train.csv && "
            "python src/pipelines/bronze_pipeline.py validation.csv && "
            "python src/pipelines/bronze_pipeline.py test.csv"
        ),
    )

    silver_pipeline_task = BashOperator(
        task_id="run_silver_pipeline",
        bash_command=(
            "cd /opt/airflow/Flights_Mlops && "
            "python src/pipelines/silver_pipeline.py train.csv && "
            "python src/pipelines/silver_pipeline.py validation.csv && "
            "python src/pipelines/silver_pipeline.py test.csv"
        ),
    )

    gold_pipeline_task = BashOperator(
        task_id="run_gold_pipeline",
        bash_command="cd /opt/airflow/Flights_Mlops && python src/pipelines/gold_pipeline.py",
    )

    fetch_data >> bronze_pipeline_task >> silver_pipeline_task >> gold_pipeline_task
