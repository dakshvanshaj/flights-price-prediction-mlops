# Orchestration with Apache Airflow (Work in Progress)

> **Note on Orchestration:** The primary method for pipeline orchestration in this project is **DVC pipelines** (defined in `dvc.yaml`). This document describes an alternative, more advanced setup using **Apache Airflow** for scheduled and production-grade workflow management. This setup is considered a future enhancement.

This document outlines the custom setup for Apache Airflow within this project, designed for clarity, maintainability, and production-readiness.

## 1. Architecture Overview

Our custom Airflow setup is containerized using Docker and orchestrated with Docker Compose. It consists of several key services that work together to provide a robust environment for orchestrating MLOps pipelines.

The core components are:
- **Custom Airflow Image**: A bespoke Docker image built from `airflow/Dockerfile`, containing Airflow, the project's source code, and all necessary dependencies.
- **PostgreSQL Database**: The metadata backend where Airflow stores information about DAGs, task instances, and connections.
- **Airflow Services**: The scheduler and webserver that form the heart of Airflow's execution engine (using `LocalExecutor` for simplicity in this setup).

## 2. Service Breakdown

The `airflow/docker-compose.yaml` file orchestrates the following services:

- **`postgres-airflow`**:
    - **Image**: `postgres:13`
    - **Purpose**: Acts as the persistent metadata database for Airflow.
- **`airflow-init`**:
    - **Image**: Custom `flights-mlops-airflow:latest`
    - **Purpose**: A one-time initialization service that sets up the database schema and creates the admin user.
- **`airflow-webserver`**:
    - **Image**: Custom `flights-mlops-airflow:latest`
    - **Purpose**: Runs the Airflow UI, accessible on `http://localhost:8080`.
- **`airflow-scheduler`**:
    - **Image**: Custom `flights-mlops-airflow:latest`
    - **Purpose**: The core service that monitors DAGs and triggers their execution.

*Note: This setup uses the `LocalExecutor`, so Celery-related services like Redis and workers are not required.*

## 3. How to Run

### Prerequisites
- Docker and Docker Compose
- An `.env` file in the `airflow` directory with `AIRFLOW_UID=$(id -u)` to ensure correct file permissions.

### Step 1: Build the Custom Image
This command builds the custom Docker image containing Airflow and your project code.

```bash
# Run from the project root
docker compose -f airflow/docker-compose.yaml build
```

### Step 2: Initialize the Database
This one-time command sets up the Airflow metadata database.

```bash
# Run from the project root
docker compose -f airflow/docker-compose.yaml up airflow-init
```

### Step 3: Start Airflow
This command starts the webserver and scheduler.

```bash
# Run from the project root
docker compose -f airflow/docker-compose.yaml up -d
```

The Airflow UI will be available at `http://localhost:8080` (default credentials: `airflow`/`airflow`).

## 4. DAG Development
With this setup, the entire project directory is available inside the Airflow containers at the `/app` path.

- **Use Relative Paths**: Access scripts from your DAGs using paths relative to the project root (e.g., `src/pipelines/bronze_pipeline.py`).
- **PYTHONPATH**: The `PYTHONPATH` is set to `/app`, so you can import custom modules directly (e.g., `from src.shared.utils import ...`).