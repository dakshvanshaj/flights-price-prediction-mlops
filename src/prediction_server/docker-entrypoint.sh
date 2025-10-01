#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ==============================================================================
# STEP 1: DVC AUTHENTICATION (BACKBLAZE B2)
# Use the unique Backblaze credentials to configure DVC.
# ==============================================================================

echo "Configuring DVC remote for Backblaze B2..."
dvc remote add -d -f "$DVC_REMOTE_NAME" s3://"$DVC_BUCKET_NAME"
dvc remote modify "$DVC_REMOTE_NAME" endpointurl "$DVC_S3_ENDPOINT_URL"
# Pass the B2 credentials directly to the DVC config commands
dvc remote modify --local "$DVC_REMOTE_NAME" access_key_id "$B2_ACCESS_KEY_ID"
dvc remote modify --local "$DVC_REMOTE_NAME" secret_access_key "$B2_SECRET_ACCESS_KEY"

echo "Pulling DVC tracked directories..."
dvc pull models
echo "DVC pull complete."

# ==============================================================================
# STEP 2: PREPARE FOR APPLICATION RUNTIME (MLFLOW)
# Set the standard AWS variables using our MLflow-specific ones. The application
# (Uvicorn/FastAPI) will inherit these and use them to talk to MLflow.
# ==============================================================================
echo "Setting up MLflow credentials for the application..."
export AWS_ACCESS_KEY_ID="${MLFLOW_AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${MLFLOW_AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${MLFLOW_AWS_DEFAULT_REGION}"
# Also export the MLflow tracking URI for the app
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"

echo "Starting application..."
# The 'exec' command replaces this script with your application.
# Your app will now run with the MLflow credentials set correctly.
exec "$@"

