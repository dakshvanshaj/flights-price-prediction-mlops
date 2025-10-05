# MLOps Tooling

This project leverages a modern stack of MLOps tools to ensure reproducibility, scalability, and maintainability. Each tool plays a specific and crucial role in the project lifecycle.

| Tool | Category | Role in Project |
| :--- | :--- | :--- |
| **GitHub** | Source Code Management | Manages the source code repository, facilitates collaboration, and hosts the CI/CD workflows via GitHub Actions. |
| **DVC (Data Version Control)** | Data & Model Versioning | Versions large data files, models, and intermediate artifacts. It works alongside Git to ensure every experiment is fully reproducible. |
| **Conda** | Environment Management | Creates isolated Python environments to ensure consistency across development and execution stages. |
| **uv** | Dependency Management | A fast Python package installer and resolver, used for virtual environment, installation and management of dependencies using `uv.lock`. |
| **DVC Pipelines** | Pipeline Orchestration | The primary tool for orchestrating the multi-stage data pipeline (`dvc.yaml`). It automatically tracks dependencies and manages execution. |
| **Great Expectations** | Data Quality & Validation | Acts as the primary data quality gate. It defines and runs "expectation suites" to validate data at the Bronze, Silver, and Gold stages. |
| **MLflow** | Experiment Tracking & Model Registry | Serves as the central hub for MLOps. It tracks experiments, logs parameters and metrics, and manages the lifecycle of trained models in the Model Registry. |
| **SHAP** | Model Explainability | Provides deep insights into model behavior by explaining the output of machine learning models, ensuring transparency and trust. |
| **FastAPI** | API Framework | Used to build the high-performance, production-ready API for serving the champion model. |
| **Docker** | Containerization | Packages the FastAPI prediction server and its dependencies into a portable container image for deployment. |
| **GitHub Actions** | CI/CD | Automates the testing, validation, and deployment pipelines, ensuring code quality and enabling seamless releases. |
| **Google Artifact Registry** | Deployment | A private Docker registry used to securely store and manage the prediction server's container images. |
| **Google Cloud Run** | Deployment | A serverless platform used to deploy and scale the containerized FastAPI prediction server. |
| **Backblaze B2** | Cloud Infrastructure | Provides S3-compatible object storage that serves as the remote backend for DVC, storing all large data and model files. |
| **AWS (EC2, RDS, S3)** | Cloud Infrastructure | A suite of AWS services used to host the remote MLflow tracking server: **EC2** for the virtual server, **RDS** for the PostgreSQL metadata database, and **S3** for the artifact store. |
| **Streamlit** | Frontend | Used as a frontend tool for prediction API. |