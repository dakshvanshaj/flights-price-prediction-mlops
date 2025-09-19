# API Reference

This document provides a detailed reference for the FastAPI prediction server.

-   **Source Code:** `src/prediction_server/`

## 1. Overview

The prediction server is a high-performance API built with FastAPI that serves the champion LightGBM model. Its primary purpose is to provide real-time flight price predictions based on input features.

The server is designed with production best practices in mind:
-   **Startup & Shutdown Logic**: It loads the model and all necessary preprocessing artifacts into memory on startup to ensure low-latency predictions and gracefully releases them on shutdown.
-   **Data Validation**: It uses Pydantic schemas to enforce a strict contract for all incoming requests and outgoing responses, preventing invalid data from causing errors.
-   **Centralized Configuration**: Key settings, like the production model name, are managed in a dedicated configuration module.

## 2. Running the API Server

To run the server locally, navigate to the project's root directory and use `uvicorn`:

If you have installed the project using pyproject.toml then

```bash
uvicorn prediction_server.main:app 
```

We can pass in other optional parameters like
```bash
uvicorn prediction_server.main:app --reload --host 0.0.0.0 --port 80
```

or directly 
```bash
fastapi run main.py
```

for development
```bash
fastapi dev main.py
```


Once running, the interactive API documentation (provided by Swagger UI) will be available at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 3. API Endpoints

### Health Check

-   **Endpoint:** `GET /`
-   **Description:** A simple health check endpoint to confirm that the API is running and responsive.
-   **Response:**
    ```json
    {
      "status": "ok",
      "message": "Welcome to the Flight Price Prediction API"
    }
    ```

### Prediction

-   **Endpoint:** `POST /prediction`
-   **Description:** The core endpoint that takes flight details, preprocesses them, runs the prediction, and returns the final estimated price.

#### Request Body

The request body must be a JSON object conforming to the following structure. The API will automatically validate the data types and constraints (e.g., `time` and `distance` must be greater than 0).

```json
{
  "from_location": "Recife (PE)",
  "to_location": "Florianopolis (SC)",
  "flight_type": "firstClass",
  "time": 1.76,
  "distance": 676.53,
  "agency": "FlyingDrops",
  "date": "2019-09-26"
}
```

#### Response Body

A successful request will return a JSON object with the predicted price.

```json
{
  "predicted_price": 1434.35
}
```

## 4. Configuration

The server's behavior is controlled by the `src/prediction_server/config_app.py` module. This file specifies which model to load from the MLflow Model Registry.

-   `MODEL_NAME`: The registered name of the model (e.g., `LGBMR_Champion`).
-   `MODEL_VERSION_ALIAS`: The alias of the model version to use (e.g., `production`).
-   `MODEL_FLAVOR`: The MLflow flavor required to load the model (e.g., `lightgbm`).

This setup allows for easy updates to the production model without changing the API's source code.

## 5. Containerization and Deployment

The prediction server is containerized using Docker to ensure a consistent and portable deployment environment. This section explains the key components involved in building and running the Docker image.

### 5.1. Dockerfile

The `Dockerfile` defines the steps to build the Docker image for the prediction server. It sets up the Python environment, installs dependencies, copies the application code, and configures the entrypoint.

```dockerfile
# Use a slim Python image for a smaller footprint
FROM python:3.12.9-slim

WORKDIR /app

# 1. Copy only the requirements file
COPY src/prediction_server/requirements.prod.txt .

# 2. Install dependencies. This layer is now cached.
RUN pip install uv && uv pip install -r requirements.prod.txt --system
# LightGBM system dependency
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 3. Now, copy the rest of the project files.
COPY . .

# 4. Set the PYTHONPATH to include the 'src' directory
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 5. Make the entrypoint script executable
RUN chmod +x /app/src/prediction_server/docker-entrypoint.sh

# 6. Set the entrypoint script to handle runtime setup
ENTRYPOINT ["/app/src/prediction_server/docker-entrypoint.sh"]

# Expose the port the app runs on
EXPOSE 8000

# 7. Set the default command to be executed by the entrypoint
CMD ["uvicorn", "prediction_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2. docker-entrypoint.sh

The `docker-entrypoint.sh` script is the entrypoint for the Docker container. Its primary role is to handle dynamic setup tasks, specifically configuring DVC and pulling necessary models, before the main application starts.

```bash
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Exit if DVC-specific environment variables are not set
if [ -z "$DVC_AWS_ACCESS_KEY_ID" ] || [ -z "$DVC_AWS_SECRET_ACCESS_KEY" ]; then
  echo "ERROR: DVC_AWS_ACCESS_KEY_ID and DVC_AWS_SECRET_ACCESS_KEY must be set."
  exit 1
fi

# Configure DVC remote with DVC-specific credentials
echo "Configuring DVC remote for S3..."
dvc remote add  -d -f "$DVC_REMOTE_NAME" s3://"$DVC_BUCKET_NAME"
dvc remote modify "$DVC_REMOTE_NAME" endpointurl "$DVC_S3_ENDPOINT_URL"
dvc remote modify  --local "$DVC_REMOTE_NAME" region "$DVC_S3_ENDPOINT_REGION"
dvc remote modify  --local "$DVC_REMOTE_NAME" access_key_id "$DVC_AWS_ACCESS_KEY_ID"
dvc remote modify  --local "$DVC_REMOTE_NAME" secret_access_key "$DVC_AWS_SECRET_ACCESS_KEY"

# Pull the DVC-tracked models
echo "Pulling DVC tracked directories..."
dvc pull models

echo "DVC pull complete. Starting application..."

# Execute the command passed to this script (e.g., uvicorn)
exec "$@"
```
### 5.3. prediction_app.ev
```env
# SAMPLE ONLY 
# ----------------------------------
# MLflow Production Configuration
# ----------------------------------
# The public IP or domain of your EC2 instance running the MLflow server.
MLFLOW_TRACKING_URI=http://PUBLICIP:5000

# ----------------------------------
# AWS Credentials for MLflow Artifacts
# ----------------------------------
# Credentials for an IAM user with read-only access to your S3 artifact bucket.
AWS_ACCESS_KEY_ID=AWSACCESSKEY
AWS_SECRET_ACCESS_KEY=AWSSECRETACCESSKEY
AWS_DEFAULT_REGION=REGION


# ---------------------------------------------------------------------------- #
#                           DVC Credentials for data                           #
# ---------------------------------------------------------------------------- #
DVC_REMOTE_NAME=remotename
DVC_BUCKET_NAME=bucketnameinaws
DVC_S3_ENDPOINT_URL=https://ENDPOINT
DVC_S3_ENDPOINT_REGION=REGION
DVC_AWS_ACCESS_KEY_ID=DVCAWSKEY
DVC_AWS_SECRET_ACCESS_KEY=DVCAWSSECRETKEY
```

**Key Functions of the Entrypoint Script:**

-   **DVC Configuration**: It dynamically configures the DVC remote storage using environment variables (`DVC_AWS_ACCESS_KEY_ID`, `DVC_AWS_SECRET_ACCESS_KEY`, `DVC_REMOTE_NAME`, `DVC_BUCKET_NAME`, `DVC_S3_ENDPOINT_URL`, `DVC_S3_ENDPOINT_REGION`). This allows the container to fetch DVC-tracked data from S3.
-   **Model Pulling**: It executes `dvc pull models` to download the necessary model artifacts into the container before the FastAPI application starts. This ensures the application has access to the latest models.
-   **Application Execution**: Finally, `exec "$@"` ensures that the command passed to `docker run` (e.g., `uvicorn prediction_server.main:app ...`) is executed as the main process within the container.

### 5.4. Building and Running the Docker Image

1.  **Build the Docker Image**
    Navigate to the project root directory (where the `Dockerfile` is located) and run:
    ```bash
    docker build -t prediction-server:latest -f prediction_server/Dockerfile .
    ```

2.  **Run the Docker Container**
    To run the container, you need to provide the DVC/AWS credentials as environment variables. It is highly recommended to use an `.env` file for this (refer to the [MLflow Integration and Deployment](MLOps/mlflow.md) documentation for details on creating `prediction_app.env`).

    ```bash
    docker run --env-file ./src/prediction_server/prediction_app.env -p 8000:8000 prediction-server:0.2
    ```

    -   `--env-file ./src/prediction_server/prediction_app.env`: Injects environment variables from the specified file into the container.
    -   `-p 8000:8000`: Maps port 8000 of the host to port 8000 inside the container.

    The API will then be accessible via `http://localhost:8000` (or the host's IP address).
