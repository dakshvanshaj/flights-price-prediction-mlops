# Flight Price Prediction MLOps Project

This project is a comprehensive, end-to-end MLOps pipeline for predicting flight prices. It leverages a modern stack of data and machine learning tools to build a reproducible, automated, and robust system that covers the entire lifecycle, from data ingestion and validation to model training, explainability, and serving.

## Features

-   **Data & Pipeline Versioning:** Uses **DVC** to version control data, models, and intermediate artifacts, ensuring full reproducibility.
-   **Data Pipeline Orchestration:** Leverages **Apache Airflow** to orchestrate the entire data processing pipeline, running inside a Docker container for portability.
-   **Automated Data Validation:** Integrates **Great Expectations** to enforce data quality and prevent bad data from moving through the pipeline.
-   **Structured Data Processing:** Implements a multi-layered data processing approach (Bronze -> Silver -> Gold) to progressively validate, clean, transform, and enrich the data.
-   **Experiment Tracking & Model Management:** Integrates **MLflow** for comprehensive experiment tracking, parameter logging, metric comparison, and model registration.
-   **Model Serving:** Includes a production-ready REST API built with **FastAPI** to serve the champion model for real-time predictions.
-   **Model Explainability:** Uses **SHAP** to provide deep insights into the champion model's behavior, ensuring transparency and trust.
-   **Reproducible Environment:** Project dependencies are managed with `pyproject.toml`, `requirements.txt` for `pip` and `environment.yaml` for `conda`, ensuring a consistent runtime environment.
-   **Automated Testing:** Includes a `tests/` directory for unit and integration tests.

## Project Structure

```
Flights_Mlops/
├── airflow/              # Airflow configuration, Dockerfile, and DAGs
├── data/                 # Raw, processed, and engineered data (tracked by DVC)
├── docs/                 # All project documentation, including the MkDocs site
├── models/               # Trained models and transformers (tracked by DVC)
├── src/                  # All Python source code
│   ├── data_ingestion/
│   ├── data_split/
│   ├── data_validation/
│   ├── gold_data_preprocessing/
│   ├── silver_data_preprocessing/
│   ├── prediction_server/ # FastAPI application for model serving
│   └── pipelines/        # Bronze, Silver, Gold, Training, and Tuning pipelines
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

## Project Documentation

This project is documented using MkDocs with the Material theme. To view the documentation locally, run the following command from the project root:

```bash
mkdocs serve -a localhost:8001
```

This will start a local server, and you can view the full documentation site by navigating to `http://localhost:8001` in your browser.

## Usage

You can run the data processing and model training pipelines using either DVC or Airflow.

### Running with DVC

DVC allows you to run the entire pipeline or specific stages. The pipeline is defined in `dvc.yaml`.
 
To run the full pipeline from data processing through training:

```bash
dvc repro
```

This will execute all stages defined in `dvc.yaml` in the correct order.

### Running with Airflow

The project includes an Airflow setup to orchestrate the data pipeline in a production-like environment.

1.  **Build and start the Airflow containers:**
    ```bash
    cd airflow
    # Build the custom Airflow image
    docker build -t airflow-flights-mlops .
    # Start the services
    docker compose up -d 
    ```

2.  **Access the Airflow UI:**
    Open your browser and go to `http://localhost:8080`. The default credentials are `airflow` / `airflow`.

3.  **Trigger the DAG:**
    In the Airflow UI, find the `data_preprocessing_pipeline` DAG and trigger it manually. This DAG will:
    -   Pull the latest data using `dvc pull`.
    -   Execute the Bronze, Silver, and Gold pipelines in sequence.

## Roadmap & Future Work

This project provides a solid foundation for a production-ready MLOps workflow. The next steps are to build out CI/CD and monitoring capabilities.

1.  **CI/CD Automation:**
    -   [ ] Set up a **GitHub Actions** workflow to automate testing, linting, and pipeline reproduction (`dvc repro`) on every push.
2.  **Monitoring:**
    -   [ ] Implement data validation checks on incoming prediction requests.
    -   [ ] Set up monitoring for data drift and model performance degradation over time.
    -   [ ] Create dashboards to visualize model performance and API health.
