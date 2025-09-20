# Use a slim Python image for a smaller footprint
FROM python:3.12.9-slim

WORKDIR /app

# 1. Install system dependencies and a faster package installer (uv)
# This is done in a single layer to reduce image size.
# libgomp1 is required by LightGBM.
RUN apt-get update && \
    apt-get install -y libgomp1  && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv


# 2. Copy only the requirements file and install dependencies
# This layer is cached and only re-runs if requirements change, speeding up builds.
COPY src/prediction_server/requirements.prod.txt ./requirements.prod.txt
RUN uv pip install -r requirements.prod.txt --system


# 3. Set the PYTHONPATH environment variable
# This allows Python to find modules inside the 'src' directory, which is
# essential because the prediction_server code imports from shared, gold, and silver.
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 4. Selectively copy only the necessary source code and configuration
# This creates a lean, secure, and cache-friendly image.
COPY src/prediction_server /app/src/prediction_server
COPY src/shared /app/src/shared
COPY src/gold_data_preprocessing /app/src/gold_data_preprocessing
COPY src/silver_data_preprocessing /app/src/silver_data_preprocessing

COPY params.yaml /app/params.yaml

# 5. Copy DVC files required by the entrypoint to pull the 'models' directory
# dvc.yaml and dvc.lock define the 'models' output from the training pipeline.
COPY dvc.yaml .
COPY dvc.lock .

# Initialize DVC
RUN dvc init --no-scm 

# 6. Make the entrypoint script executable
# The script is copied in step 4, so we reference its final location.
RUN chmod +x /app/src/prediction_server/docker-entrypoint.sh

# 7. Set the entrypoint script to handle runtime setup
ENTRYPOINT ["/app/src/prediction_server/docker-entrypoint.sh"]

# 8. Expose the port the app runs on
EXPOSE 8000

# 9. Set the default command to be executed by the entrypoint
# This starts the uvicorn server after the entrypoint script finishes.
CMD ["uvicorn", "prediction_server.main:app", "--host", "0.0.0.0", "--port", "8000"]