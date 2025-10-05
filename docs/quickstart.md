# ✈️ Flight Price Prediction MLOps Project

This project is a comprehensive, end-to-end MLOps pipeline for predicting flight prices. It leverages a modern stack of data and machine learning tools to build a reproducible, automated, and robust system that covers the entire lifecycle, from data ingestion and validation to model training, explainability, and serving.

## ✨ Features

-   **Data & Pipeline Versioning:** Uses **[DVC](docs/MLOps/dvc.md)** to version control data, models, and intermediate artifacts, ensuring full reproducibility.
-   **Declarative Pipeline Orchestration:** The entire ML pipeline is defined as code in **[`dvc.yaml`](docs/MLOps/dvc_pipeline.md)**, allowing for robust, dependency-aware execution.
-   **Automated Data Validation:** Integrates **[Great Expectations](docs/Data%20Pipelines/data_pipelines.md)** at each pipeline stage to enforce data quality.
-   **Experiment Tracking & Model Management:** Integrates **[MLflow](docs/MLOps/mlflow.md)** for comprehensive experiment tracking, parameter logging, and model registration.
-   **Automated CI/CD**: Implements **[GitHub Actions](docs/MLOps/ci.md)** for automated linting, testing, pipeline validation, and **[deployment](docs/MLOps/cd.md)** to Google Cloud.
-   **Production-Ready API**: Includes a high-performance **[FastAPI](docs/API/api_reference.md)** server to serve the champion model, containerized with **[Docker](docs/MLOps/docker.md)**.
-   **Reproducible Environment:** Project dependencies are managed with **`uv`** and locked in `requirements.lock` for fast, deterministic setups.


## 🖥️ User Interface

This project includes an interactive web application built with **Streamlit** that serves as a user-friendly interface for the flight price prediction API.

![Streamlit UI Screenshot](docs/img/streamlit_frontend.png)

### Running the Frontend

1.  **Ensure the API is running** either locally via Docker or as a deployed service.
2.  **Configure the API URL** by creating a `.env` file in the `frontend_streamlit/` directory.
    ```ini
    # frontend_streamlit/.env
    API_URL=http://127.0.0.1:9000/prediction
    ```
3.  **Run the Streamlit app** from the project root:
    ```bash
    streamlit run frontend_streamlit/app.py
    ```
*For more details, see the [Frontend Documentation](docs/frontend.md).*

## 🚀 Quickstart: Local Setup

Follow these steps to get the project running on your local machine.

### Prerequisites

-   Python 3.12+
-   [uv](https://github.com/astral-sh/uv): An extremely fast Python package installer and resolver.
-   [Git](https://git-scm.com/)
-   [DVC](https://dvc.org/doc/install)
-   [act](https://github.com/nektos/act) (Optional, for local CI/CD testing)

### 1. Clone the Repository

```bash
git clone https://github.com/dakshvanshaj/flights-price-prediction-mlops.git
cd flights-price-prediction-mlops
```

### 2. Create Virtual Environment & Install Dependencies

```bash
# Create and activate a virtual environment using uv
uv venv
source .venv/bin/activate
# On Windows: .\.venv\Scripts\activate

# Sync the environment with the lock file for a reproducible setup
uv sync --all-extras --locked

# Install the project in editable mode
uv pip install -e .
```

### 3. Get the Project Data

You have two options to get the data needed to run the pipelines.

#### Option A: Quick Local Start (No Credentials Needed)

This is the fastest way to get started. This project includes a Git-tracked archive with the initial raw data.

```bash
# Unzip the archive to get the initial flights.csv
unzip data/archieve-git-tracked/raw.zip -d data/raw/
```

#### Option B: Full DVC Setup (Recommended)

To get all versioned data, models, and artifacts, you must configure DVC to connect to the remote S3-compatible storage. See the [**DVC Integration Guide &raquo;**](docs/MLOps/dvc.md) for more details.

```bash
# Configure the DVC remote endpoint URL and credentials.
dvc remote add -d myremote s3://your-bucket-name
dvc remote modify --local myremote endpointurl <YOUR_S3_ENDPOINT_URL>
dvc remote modify --local myremote access_key_id <YOUR_ACCESS_KEY_ID>
dvc remote modify --local myremote secret_access_key <YOUR_SECRET_ACCESS_KEY>

# Pull all DVC-tracked data and model artifacts
dvc pull -v
```

### 4. Set Up MLflow Tracking Server (Optional)

By default, MLflow will log experiments locally. To use a remote, centralized server, create a `.env` file in the project root and populate it with your server's credentials. The application will automatically load these using `dotenv`.

```ini
# .env file
MLFLOW_TRACKING_URI=http://your-remote-mlflow-server-ip:5000
MLFLOW_AWS_ACCESS_KEY_ID=your_mlflow_s3_access_key
MLFLOW_AWS_SECRET_ACCESS_KEY=your_mlflow_s3_secret_key
MLFLOW_AWS_DEFAULT_REGION=your_s3_bucket_region
```

*For a complete guide on deploying a production-grade MLflow server, see the [**MLflow Deployment Documentation &raquo;**](docs/MLOps/mlflow.md).*

### 5. Running the Pipelines

You can run the project's pipelines in several ways. See the [**DVC Pipeline Documentation &raquo;**](docs/MLOps/dvc_pipeline.md) for a full breakdown.

#### Method 1: Automated DAG Execution with DVC (Recommended)

```bash
# Run the entire pipeline from start to finish
dvc repro

# Force if it shows no change in pipeline
dvc repro -f

# Or, run the pipeline up to a specific stage
dvc repro gold_pipeline

# Alternatively using experiment tracking
dvc exp run

dvc exp show
```

#### Method 2: Manual Script Execution (For Debugging)

Use the CLI shortcuts defined in `pyproject.toml`:

```bash
run-bronze-pipeline train.csv
run-silver-pipeline train.csv
run-gold-pipeline
run-training-pipeline
```

### 🔧 Configuring the Pipelines

The behavior of the pipelines can be customized without changing the source code.

-   **High-Level Parameters (`params.yaml`)**: Control the overall strategy, such as which model to run (`model_config_to_run`) or whether to use the tree-based preprocessing path (`is_tree_model`).
-   **Low-Level Configuration (`src/shared/config/`)**: Contains static configurations like file paths and column lists for transformations.

## 🤖 Local CI/CD Testing with `act`

You can run the GitHub Actions workflows locally using [act](https://github.com/nektos/act). This is incredibly useful for testing changes to your CI/CD pipeline without pushing to GitHub. See the [**CI**](docs/MLOps/ci.md) and [**CD**](docs/MLOps/cd.md) docs for more details.

### Setup

Create a `.secrets` file in the project root and populate it with the necessary credentials for local testing.

> **Warning:** The `.secrets` file contains sensitive information. It is already listed in `.gitignore` and should **never** be committed to version control.

### Usage

```bash
# Run the default `on: push` workflow
act

# Run a specific job from a workflow
act -j test_and_lint

# Run the CD workflow by simulating a tag push
act push -W .github/workflows/cd.yml -e tag_push_event.json
```

## 📚 Full Project Documentation

This project is documented using MkDocs. To view the full, searchable documentation site locally, run:

```bash
mkdocs serve
```

Navigate to `http://127.0.0.1:8000` in your browser.