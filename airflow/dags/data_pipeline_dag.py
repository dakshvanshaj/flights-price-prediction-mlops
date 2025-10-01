import pendulum
import subprocess
import os
import logging

from airflow.decorators import dag, task

# Get a logger
logger = logging.getLogger(__name__)

# Define constants for paths and commands to keep the code clean
# The project root inside the Airflow container is /opt/airflow
PROJECT_ROOT = "/opt/airflow"
DVC_PULL_COMMAND = "dvc pull -v"

SPLIT_DATA_SCRIPT = "src/data_split/split_data.py"
BRONZE_SCRIPT = "src/pipelines/bronze_pipeline.py"
SILVER_SCRIPT = "src/pipelines/silver_pipeline.py"
GOLD_SCRIPT = "src/pipelines/gold_pipeline.py"
TRAINING_SCRIPT = "src/pipelines/training_pipeline.py"
DATA_SPLITS = ["train", "validation", "test"]

default_args = {
    "owner": "Daksh",
    "retries": 0,
    "doc_md": """
    ### DVC Pipeline Orchestration DAG (Refactored with .expand())
    This DAG replicates the workflow defined in `dvc.yaml` using modern Airflow features.
    - **fetch_data_with_dvc**: Pulls data and models using DVC.
    - **split_data**: Splits the raw data into train, validation, and test sets.
    - **process_bronze_to_silver**: A dynamic task that runs the bronze and silver steps in parallel for each data split.
    - **run_gold_pipeline**: Consolidates silver data and runs the gold feature engineering pipeline.
    - **run_training_pipeline**: Trains the model on the final gold data.
    """,
}


def run_command(command: str, cwd: str):
    """
    Helper function to run shell commands, log output, and handle errors.
    """
    logger.info(f"Running command: '{command}' in directory '{cwd}'")
    process = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    if process.stdout:
        logger.info("--- STDOUT ---\n%s", process.stdout)
    if process.stderr:
        logger.error("--- STDERR ---\n%s", process.stderr)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=command,
            output=process.stdout,
            stderr=process.stderr,
        )
    logger.info(f"Command '{command}' executed successfully.")


@dag(
    dag_id="dvc_pipeline_orchestration",
    start_date=pendulum.datetime(2025, 8, 1, tz="UTC"),
    description="Orchestrates the full DVC pipeline for data processing and model training.",
    schedule=None,
    catchup=False,
    tags=["dvc", "mlops", "pipeline"],
    default_args=default_args,
)
def dvc_pipeline_orchestration_dag():
    """
    This DAG replicates the structure of dvc.yaml to orchestrate the ML pipeline.
    It uses .expand() to create a more efficient, parallel workflow.
    """

    @task
    def fetch_data_with_dvc():
        """Pulls the latest data using DVC."""
        # DVC needs to be explicitly configured inside the task environment.
        # This logic mirrors the setup in src/prediction_server/docker-entrypoint.sh
        logger.info("Configuring DVC remote...")

        # Use os.environ.get() for safer access to environment variables
        remote_name = os.environ.get("DVC_REMOTE_NAME")
        bucket_name = os.environ.get("DVC_BUCKET_NAME")
        endpoint_url = os.environ.get("DVC_S3_ENDPOINT_URL")
        region = os.environ.get("DVC_S3_ENDPOINT_REGION")
        access_key = os.environ.get("DVC_AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("DVC_AWS_SECRET_ACCESS_KEY")

        # Run each command separately for better logging and error isolation
        run_command(
            f"dvc remote add -d -f {remote_name} s3://{bucket_name}",
            PROJECT_ROOT,
        )
        run_command(
            f"dvc remote modify {remote_name} endpointurl {endpoint_url}",
            PROJECT_ROOT,
        )
        run_command(
            f"dvc remote modify --local {remote_name} region {region}",
            PROJECT_ROOT,
        )
        run_command(
            f"dvc remote modify --local {remote_name} access_key_id {access_key}",
            PROJECT_ROOT,
        )
        run_command(
            f"dvc remote modify --local {remote_name} secret_access_key {secret_key}",
            PROJECT_ROOT,
        )

        logger.info("DVC configured. Pulling data...")
        run_command(DVC_PULL_COMMAND, PROJECT_ROOT)

    @task
    def split_data():
        """Splits the raw data into train/validation/test sets."""
        run_command(f"python {SPLIT_DATA_SCRIPT}", PROJECT_ROOT)

    @task
    def process_bronze_to_silver(file_prefix: str):
        """Runs the bronze and silver pipelines sequentially for a single data split."""
        logger.info(f"--- Processing {file_prefix} split: Bronze Stage ---")
        run_command(f"python {BRONZE_SCRIPT} {file_prefix}.csv", PROJECT_ROOT)
        logger.info(f"--- Processing {file_prefix} split: Silver Stage ---")
        run_command(f"python {SILVER_SCRIPT} {file_prefix}.csv", PROJECT_ROOT)

    @task
    def run_gold_pipeline(upstream_dependency):
        """Runs the gold pipeline after all silver tasks are complete."""
        run_command(f"python {GOLD_SCRIPT}", PROJECT_ROOT)

    @task
    def run_training_pipeline():
        """Runs the training pipeline using the correct arguments from dvc.yaml."""
        command = f"python {TRAINING_SCRIPT} train.parquet validation.parquet --test_file_name test.parquet"
        run_command(command, PROJECT_ROOT)

    # --- Define Task Dependencies ---
    fetch_op = fetch_data_with_dvc()
    split_op = split_data()

    # Dynamically create parallel tasks for bronze and silver processing
    bronze_silver_op = process_bronze_to_silver.expand(file_prefix=DATA_SPLITS)

    # The gold task will only run after all parallel bronze/silver tasks are done
    gold_op = run_gold_pipeline(upstream_dependency=bronze_silver_op)

    training_op = run_training_pipeline()

    fetch_op >> split_op >> bronze_silver_op >> gold_op >> training_op


dvc_pipeline_orchestration_dag()
