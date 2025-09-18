#!/bin/bash
set -e

# Exit if DVC-specific environment variables are not set
if [ -z "$DVC_AWS_ACCESS_KEY_ID" ] || [ -z "$DVC_AWS_SECRET_ACCESS_KEY" ]; then
  echo "ERROR: DVC_AWS_ACCESS_KEY_ID and DVC_AWS_SECRET_ACCESS_KEY must be set."
  exit 1
fi

# Configure DVC remote with DVC-specific credentials
echo "Configuring DVC remote for S3..."
dvc remote add -d "$DVC_REMOTE_NAME" s3://"$DVC_BUCKET_NAME"
dvc remote modify "$DVC_REMOTE_NAME" endpointurl "$DVC_S3_ENDPOINT_URL"
dvc remote modify  --local access_key_id "$DVC_AWS_ACCESS_KEY_ID"
dvc remote modify  --local secret_access_key "$DVC_AWS_SECRET_ACCESS_KEY"

# Pull the DVC-tracked models
echo "Pulling DVC tracked directories..."
dvc pull models

echo "DVC pull complete. Starting application..."

# Execute the command passed to this script (e.g., uvicorn)
exec "$@"
