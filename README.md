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

## Data Validation and EDA

-   **Great Expectations:** The data validation results are compiled into user-friendly HTML docs. After running the pipeline, you can find them here:
    -   `src/data_validation/great_expectations/gx/uncommitted/data_docs/local_site/index.html`

-   **Exploratory Data Analysis (EDA):** An comprehensive EDA and automated EDA report for the training dataset is available at:
    -   `Notebooks/eda/flights_training_eda.ipynb`
    -   `reports/eda/flights_training_eda_report.html`


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

This project provides a solid foundation for a production-ready MLOps workflow. The next steps are to build out the model training, deployment, and monitoring capabilities.

1.  **Model Training & Experimentation:**
    -   [ ] Implement a `dvc` versioned stage for training regression models (e.g., XGBoost, LightGBM).
    -   [ ] Integrate **MLflow** for experiment tracking, parameter logging, and metric comparison.
2.  **CI/CD Automation:**
    -   [ ] Set up a **GitHub Actions** workflow to automate testing, linting, and pipeline reproduction (`dvc repro`) on every push.
3.  **Model Serving:**
    -   [ ] Create a REST API using **FastAPI** to serve the best-performing model.
    -   [ ] Containerize the serving application with **Docker** for easy deployment.
4.  **Monitoring:**
    -   [ ] Implement data validation checks on incoming prediction requests.
    -   [ ] Set up monitoring for data drift and model performance degradation over time.
