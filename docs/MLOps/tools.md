# MLOps Tooling

This project leverages a modern stack of MLOps tools to ensure reproducibility, scalability, and maintainability. Each tool plays a specific and crucial role in the lifecycle of the machine learning model.

| Tool | Category | Role in Project |
| :--- | :--- | :--- |
| **Conda** | Environment Management | Manages Python dependencies and creates isolated, reproducible environments to ensure consistency across development and execution stages. |
| **DVC (Data Version Control)** | Data & Model Versioning | Versions large data files, models, and intermediate artifacts alongside the Git repository, ensuring that every experiment is fully reproducible. |
| **Great Expectations** | Data Quality & Validation | Acts as the primary data quality gate. It defines and runs "expectation suites" to validate data at the Bronze, Silver, and Gold stages, preventing bad data from propagating. |
| **MLflow** | Experiment Tracking & Model Registry | Serves as the central hub for MLOps. It tracks experiments, logs parameters, metrics, and artifacts, and manages the lifecycle of trained models in the Model Registry. |
| **Apache Airflow** | Pipeline Orchestration | Orchestrates the execution of the data and machine learning pipelines. It is responsible for running the Bronze, Silver, and Gold data pipelines in the correct sequence. |
| **Docker** | Containerization | Containerizes the Airflow environment, ensuring that the orchestrator and its dependencies are portable and run consistently across different systems. |
| **GitHub** | Source Code Management | Manages the source code repository, facilitates collaboration, and serves as the foundation for future CI/CD automation with GitHub Actions. |
| **SHAP (SHapley Additive exPlanations)** | Model Explainability | Provides deep insights into model behavior by explaining the output of machine learning models, ensuring transparency and trust in the predictions. |

---

## Future Work

The following tools are planned for future integration to further enhance the MLOps capabilities of this project:

-   **GitHub Actions**: To build a CI/CD pipeline for automated testing, validation, and deployment.
-   **FastAPI**: To create a high-performance API endpoint for serving the champion model in a production environment.
-   **Docker**: To expand its use beyond Airflow to containerize the model serving application, creating a portable and scalable deployment artifact.
