# 🐳 Docker Integration

This document outlines the project's containerization strategy, explaining how Docker is used to create consistent, portable, and production-ready environments for different components of the MLOps lifecycle.

## 🎯 1. Overview of Docker Strategy

Docker is a cornerstone of this project, used to solve two primary challenges:

1.  **Orchestration Environment `(WIP)`**: It provides a self-contained, reproducible environment for running the **Apache Airflow** scheduler and webserver, ensuring that the data pipeline orchestration is consistent across all systems.
2.  **Deployment Environment**: It packages the **FastAPI prediction server** into a lightweight, secure, and portable image that can be deployed anywhere—from a local machine to a serverless cloud platform like Google Cloud Run.

## ⚙️ 2. Use Case 1: Orchestration with Airflow(`WIP`)

The project includes a complete setup to run Airflow within Docker for orchestrating the data pipelines.

-   **Setup**: The environment is defined in `airflow/docker-compose.yaml`, which orchestrates the necessary Airflow services (scheduler, webserver, postgres metadata database).
-   **Custom Image**: A custom image is built via `airflow/Dockerfile` to install project-specific dependencies.

> For a detailed guide on the Airflow setup and how to run the orchestration DAGs, please see the [**Airflow Documentation**](airflow.md).

## 🚀 3. Use Case 2: Serving the Prediction API

Containerizing the prediction server is critical for deployment. Our strategy focuses on creating a minimal, secure, and efficient production image.

### Dockerfile Strategy: Multi-Stage Builds

The `src/prediction_server/Dockerfile` uses a **multi-stage build** to create a lean final image. This is a best practice that significantly reduces the image size and attack surface.

1.  **The `builder` Stage**: This first stage installs `uv` and uses it to download and compile all Python dependencies into a virtual environment. This stage contains all the build tools and cache, making it large.
2.  **The `final` Stage**: This second stage starts from a fresh, slim Python base image. It copies *only* the installed packages from the `builder` stage and the application source code. It does **not** include `uv`, build tools, or any intermediate layers, resulting in a much smaller and more secure production image.

### Entrypoint Script Strategy: Dynamic Runtime Configuration

The container's startup is managed by the `src/prediction_server/docker-entrypoint.sh` script. This script runs *before* the FastAPI application and is responsible for preparing the runtime environment:

1.  **DVC Configuration**: It uses environment variables to configure DVC to connect to the remote S3 storage.
2.  **Artifact Pulling**: It runs `dvc pull` to download the required model and data transformer artifacts into the container.
3.  **MLflow Configuration**: It sets the necessary environment variables for the application to communicate with the MLflow tracking server.
4.  **Application Execution**: Finally, it executes the main `uvicorn` command to start the FastAPI server.

This approach decouples the static container image from the dynamic runtime configuration, making the image more portable.

### Quick Commands

-   **Build the Image**:
    ```bash
    docker build -t prediction-server:latest -f src/prediction_server/Dockerfile .
    ```

-   **Run the Container**:
    ```bash
    docker run --env-file ./src/prediction_server/prediction_app.env -p 9000:9000 prediction-server:latest
    ```

> For a line-by-line breakdown of the `Dockerfile`, the `docker-entrypoint.sh` script, and a detailed guide on deploying to the cloud, please refer to the [**API Reference & Deployment Guide**](../API/api_reference.md#🐳-containerization--execution).