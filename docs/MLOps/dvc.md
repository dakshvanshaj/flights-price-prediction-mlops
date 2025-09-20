# DVC (Data Version Control) Integration

This document details the role of DVC in managing and versioning large data files and models within the project, ensuring reproducibility and collaboration.

## 1. Overview and Purpose

DVC (Data Version Control) extends Git's capabilities to handle large files, datasets, and machine learning models. In this project, DVC is used to:

-   **Version Data and Models**: Track changes to datasets and trained models alongside the code, ensuring that every experiment is fully reproducible.
-   **Enable Reproducibility**: Allows switching between different versions of data and models by simply checking out a Git commit and running `dvc pull`.
-   **Manage Large Files**: Keeps large files out of the Git repository, storing them efficiently in remote storage (like AWS S3) while Git tracks only small `.dvc` metadata files.
-   **Facilitate Collaboration**: Provides a streamlined way for team members to work with consistent versions of data and models.

## 2. DVC Setup and Configuration

### 2.1. Initial Setup

After initializing a Git repository, DVC is initialized in the project root:

```bash
dvc init
```

The primary configuration involves setting up a remote storage location, which in this project is an Blackblaze b2 S3 bucket.

### 2.2. Dynamic Remote Configuration

For production and containerized environments, this project uses a dynamic approach to configure the DVC remote storage at runtime. This is handled by the `docker-entrypoint.sh` script and relies on environment variables.

The script executes the following commands to configure the DVC remote:

```bash
# Add the remote storage location
dvc remote add -d -f "$DVC_REMOTE_NAME" s3://"$DVC_BUCKET_NAME"

# Modify the remote with specific credentials and settings
dvc remote modify "$DVC_REMOTE_NAME" endpointurl "$DVC_S3_ENDPOINT_URL"
dvc remote modify --local "$DVC_REMOTE_NAME" region "$DVC_S3_ENDPOINT_REGION"
dvc remote modify --local "$DVC_REMOTE_NAME" access_key_id "$DVC_AWS_ACCESS_KEY_ID"
dvc remote modify --local "$DVC_REMOTE_NAME" secret_access_key "$DVC_AWS_SECRET_ACCESS_KEY"
```

### 2.3. Environment Variables for Configuration

DVC requires credentials to access the S3 bucket. These are provided via the following environment variables:

-   `DVC_REMOTE_NAME`: The name to assign to the DVC remote (e.g., `myremote`).
-   `DVC_BUCKET_NAME`: The name of the S3 bucket where data and models are stored.
-   `DVC_S3_ENDPOINT_URL`: The S3 endpoint URL (optional, for custom S3-compatible storage).
-   `DVC_S3_ENDPOINT_REGION`: The AWS region of the S3 bucket.
-   `DVC_AWS_ACCESS_KEY_ID`: The AWS access key ID.
-   `DVC_AWS_SECRET_ACCESS_KEY`: The AWS secret access key.

These variables are crucial for DVC operations, especially in containerized environments. Refer to the [MLflow Integration and Deployment - Production Best Practice: Using an Environment File](mlflow.md#13-production-best-practice-using-an-environment-file) documentation for details on securely managing these credentials via `.env` files.

## 3. Core Workflow

### 3.1. Versioning Data and Models

To version a file or directory with DVC:

```bash
dvc add data/raw/flights.csv
dvc add models/
```

This command replaces the actual file/directory with a small `.dvc` file. This `.dvc` file is then tracked by Git:

```bash
git add data/raw/flights.csv.dvc models.dvc
git commit -m "Add DVC-tracked raw data and models"
```

To push the actual data to remote storage:

```bash
dvc push
```

### 3.2. Retrieving Data and Models

To retrieve the data and models associated with your current Git commit, use `dvc pull`:

```bash
dvc pull
```

This command downloads the data from the DVC remote to your local machine.

### 3.3. Reproducing Past States

One of DVC's most powerful features is the ability to reproduce past experiments by checking out a specific Git commit and retrieving the exact data and models used at that time:

```bash
git checkout <commit_hash>
dvc pull
```

This ensures full reproducibility of any experiment or model version.

## 4. DVC in the Project

### 4.1. DVC in the Docker Entrypoint

For the prediction server, DVC plays a critical role during container startup. The `docker-entrypoint.sh` script is configured to pull the necessary models before the FastAPI application launches. This ensures the server always starts with the correct model version. For the full script and its context, refer to the [API Reference - Containerization and Deployment](../API/api_reference.md#5-containerization-and-deployment) section.

### 4.2. DVC Pipelines

This project also utilizes DVC pipelines to define and manage the data processing and model training workflows. The entire end-to-end pipeline is defined in `dvc.yaml`.

To run the full DVC pipeline:

```bash
dvc repro
```

This command executes all stages defined in `dvc.yaml` in the correct order, ensuring that data dependencies are met and only necessary stages are re-executed when inputs change.