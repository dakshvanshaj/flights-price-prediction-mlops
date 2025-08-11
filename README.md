# Flight Price Prediction MLOps Project

This project is a comprehensive MLOps pipeline for predicting flight prices. It leverages a modern stack of data and machine learning tools to build a reproducible, automated, and robust system for data processing and feature engineering and flights price prediction(WIP).

## Features

-   **Data & Pipeline Versioning:** Uses **DVC** to version control data, models, and intermediate artifacts, ensuring full reproducibility.
-   **Data Pipeline Orchestration:** Leverages **Apache Airflow** to orchestrate the entire data processing pipeline, running inside a Docker container for portability.
-   **Automated Data Validation:** Integrates **Great Expectations** to enforce data quality and prevent bad data from moving through the pipeline.
-   **Structured Data Processing:** Implements a multi-layered data processing approach (Bronze -> Silver -> Gold) to progressively validate, clean, transform, and enrich the data.
-   **Reproducible Environment:** Project dependencies are managed with `pyproject.toml`, `requirements.txt` for `pip` and `environment.yaml` for `conda`, ensuring a consistent runtime environment.
-   **Automated Testing:** Includes a `tests/` directory for unit and integration tests.

## Project Structure

```
Flights_Mlops/
├── airflow/              # Airflow configuration, Dockerfile, and DAGs
├── data/                 # Raw, processed, and engineered data (tracked by DVC)
├── models/               # Trained models and transformers (tracked by DVC)
├── src/                  # All Python source code
│   ├── data_ingestion/
│   ├── data_split/
│   ├── data_validation/
│   ├── gold_data_preprocessing/
│   ├── silver_data_preprocessing/
│   └── pipelines/        # Bronze, Silver, and Gold data pipelines
├── tests/                # Unit and integration tests
├── dvc.yaml              # DVC pipeline definitions
├── params.yaml           # Parameters for DVC pipelines
├── pyproject.toml        # Project metadata and dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites

-   Python 3.12+
-   Conda (recommended for environment management)
-   Docker and Docker Compose (for running Airflow)
-   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Flights_Mlops
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate flights-mlops
    ```

3.  **Install the project in editable mode:**
    This makes the `src` code available as a package and installs the dependencies from `pyproject.toml`.
    ```bash
    pip install -e .
    ```

4.  **Pull DVC data:**
    This will download the data, models, and other artifacts tracked by DVC.
    ```bash
    dvc pull -v
    ```

## Usage

You can run the data processing pipelines using either DVC or Airflow.

### Running with DVC

DVC allows you to run the entire pipeline or specific stages. The pipeline is defined in `dvc.yaml`.

To run the full pipeline:

```bash
dvc repro
```

This will execute all stages defined in `dvc.yaml` in the correct order.

### Running with Airflow

The project includes an Airflow setup to orchestrate the pipeline in a production-like environment.

1.  **Build and start the Airflow containers:**
    ```bash
    cd airflow/
    docker-compose up --build -d
    ```

2.  **Access the Airflow UI:**
    Open your browser and go to `http://localhost:8080`.

3.  **Trigger the DAG:**
    In the Airflow UI, find the `data_preprocessing_pipeline` DAG and trigger it manually. This DAG will:
    -   Pull the latest data using `dvc pull`.
    -   Execute the Bronze, Silver, and Gold pipelines in sequence.

## Under Development

This project provides a solid foundation for an MLOps workflow. Here are some areas for future development:

-   **Model Training Pipeline:** Implement a DVC stage for training various regression models (e.g., Linear Regression, XGBoost, LightGBM) on the gold data.
-   **Model Evaluation & Selection:** Add a pipeline for model evaluation, hyperparameter tuning, and selecting the best-performing model.
-   **Model Registry:** Integrate a model registry (like MLflow ) to track experiments and manage model versions.
-   **CI/CD Integration:** Set up a CI/CD pipeline (e.g., using GitHub Actions) to automatically run tests, linting, and pipeline execution on code changes.
-   **Model Serving:** Create a REST API (e.g., using FastAPI) to serve the trained model for real-time predictions.
-   **Monitoring:** Implement monitoring for data drift, model performance, and pipeline health.
