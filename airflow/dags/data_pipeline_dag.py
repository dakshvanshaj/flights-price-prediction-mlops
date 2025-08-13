import logging
import pendulum
import subprocess

from airflow.sdk import dag, task

# Define constants for paths and commands to keep the code clean
FLIGHTS_MLOPS_DIR = "/opt/airflow/Flights_Mlops"
DVC_PULL_COMMAND = "dvc pull -v"
BRONZE_SCRIPT = "src/pipelines/bronze_pipeline.py"
SILVER_SCRIPT = "src/pipelines/silver_pipeline.py"
GOLD_SCRIPT = "src/pipelines/gold_pipeline.py"
DATA_FILES = ["train.csv", "validation.csv", "test.csv"]

default_args = {
    "owner": "Daksh",
    "retries": 0,
    "doc_md": """
    ### Data Preprocessing Pipeline 
    This DAG uses the TaskFlow API to orchestrate a DVC pull followed by
    a multi-stage data processing pipeline (Bronze, Silver, Gold).
    - **fetch_data_with_dvc**: Pulls data using DVC.
    - **run_bronze_pipeline**: Processes raw data into the bronze layer for data validation.
    - **run_silver_pipeline**: Cleans and transforms bronze data into the silver layer.
    - **run_gold_pipeline**: Aggregates silver data into the final gold layer for modeling.
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
def data_preprocessing_pipeline():
    """
    This is the main function that defines the DAG's workflow.
    Airflow will parse this function to create the DAG structure.
    """

    logger = logging.getLogger(__name__)

    def run_command(command: str, cwd: str):
        """Helper function to run shell commands and handle errors."""
        logger.info(f"Running command: '{command}' in directory '{cwd}'")
        process = subprocess.run(
            command,
            shell=True,
            check=True,  # This will raise a CalledProcessError if the command fails
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("--- STDOUT ---")
        logger.info(process.stdout)
        if process.stderr:
            logger.warning("--- STDERR ---")
            logger.warning(process.stderr)
        logger.info(f"Command '{command}' executed successfully.")
        return process.stdout  #  return output if needed

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

    # --- Define Dependencies ---
    # Even with TaskFlow, you can explicitly set dependencies.
    # This is necessary here because the tasks don't pass data to each other directly.
    # They operate on files within a shared directory.
    fetch_task = fetch_data_with_dvc()
    bronze_task = run_bronze_pipeline()
    silver_task = run_silver_pipeline()
    gold_task = run_gold_pipeline()

    fetch_task >> bronze_task >> silver_task >> gold_task


# This final call creates the DAG object that Airflow can discover.
data_preprocessing_pipeline()
