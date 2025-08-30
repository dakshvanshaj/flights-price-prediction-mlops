import logging
import pendulum
import subprocess
import os

from typing import Optional, Dict
from airflow.sdk import dag, task

# Define constants for paths and commands to keep the code clean
FLIGHTS_MLOPS_DIR = "/opt/airflow/Flights_Mlops"
DVC_PULL_COMMAND = "dvc pull -v"
BRONZE_SCRIPT = "src/pipelines/bronze_pipeline.py"
SILVER_SCRIPT = "src/pipelines/silver_pipeline.py"
GOLD_SCRIPT = "src/pipelines/gold_pipeline.py"
TRAINING_SCRIPT = "src/pipelines/training_pipeline.py"
DATA_FILES = ["train.csv", "validation.csv", "test.csv"]

default_args = {
    "owner": "Daksh",
    "retries": 0,
    "doc_md": """
    ### Data Preprocessing Pipeline
    This DAG uses the TaskFlow API to orchestrate a DVC pull followed by
    a multi-stage data processing pipeline (Bronze, Silver, Gold, and Training).
    - **fetch_data_with_dvc**: Pulls data using DVC.
    - **run_bronze_pipeline**: Processes raw data into the bronze layer.
    - **run_silver_pipeline**: Cleans bronze data into the silver layer.
    - **run_gold_pipeline**: Aggregates silver data into the gold layer.
    - **run_training_pipeline**: Trains the model on the gold data.
    """,
}


@dag(
    dag_id="data_preprocessing_pipeline_taskflow",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    description="Fetches DVC data and runs the full ML pipeline using TaskFlow API.",
    schedule=None,
    catchup=False,
    tags=["data", "preprocessing", "pipeline", "dvc", "taskflow"],
    default_args=default_args,
)
def data_preprocessing_pipeline_taskflow():
    """
    This is the main function that defines the DAG's workflow.
    Airflow will parse this function to create the DAG structure.
    """
    logger = logging.getLogger(__name__)

    def run_command(command: str, cwd: str, env: Optional[Dict[str, str]] = None):
        """
        Helper function to run shell commands, log output, and handle errors.
        """
        logger.info(f"Running command: '{command}' in directory '{cwd}'")

        # Create a full environment for the subprocess to run in.
        # This inherits the Airflow worker's environment and can be updated.
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        process = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            text=True,
        )

        # Always log stdout and stderr for better debugging
        if process.stdout:
            logger.info("--- STDOUT ---")
            logger.info(process.stdout)
        if process.stderr:
            # Log stderr as an error to make it more visible in Airflow logs
            logger.error("--- STDERR ---")
            logger.error(process.stderr)

        # Now, check if the command failed and raise an exception if it did
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode,
                cmd=command,
                output=process.stdout,
                stderr=process.stderr,
            )

        logger.info(f"Command '{command}' executed successfully.")
        return process.stdout

    @task
    def fetch_data_with_dvc():
        """Pulls the latest data using DVC."""
        run_command(DVC_PULL_COMMAND, FLIGHTS_MLOPS_DIR)

    @task
    def run_bronze_pipeline():
        """Runs the bronze data processing script for all data files."""
        for file in DATA_FILES:
            command = f"python {BRONZE_SCRIPT} {file}"
            run_command(command, FLIGHTS_MLOPS_DIR)

    @task
    def run_silver_pipeline():
        """Runs the silver data processing script for all data files."""
        for file in DATA_FILES:
            command = f"python {SILVER_SCRIPT} {file}"
            run_command(command, FLIGHTS_MLOPS_DIR)

    @task
    def run_gold_pipeline():
        """Runs the final gold data processing script."""
        run_command(f"python {GOLD_SCRIPT}", FLIGHTS_MLOPS_DIR)

    @task
    def run_training_pipeline():
        """Runs the training pipeline."""
        # Define the command and environment separately for clarity and security.
        command = (
            f"python {TRAINING_SCRIPT} train.parquet validation.parquet test.parquet"
        )

        run_command(command, FLIGHTS_MLOPS_DIR)

    # --- Define Dependencies ---
    fetch_task = fetch_data_with_dvc()
    bronze_task = run_bronze_pipeline()
    silver_task = run_silver_pipeline()
    gold_task = run_gold_pipeline()
    training_task = run_training_pipeline()

    fetch_task >> bronze_task >> silver_task >> gold_task >> training_task


data_preprocessing_pipeline_taskflow()
