# Testing Strategy

This document outlines the comprehensive testing strategy employed in this MLOps project to ensure the reliability, correctness, and robustness of the data pipelines, models, and application components.

## 1. Overview and Philosophy

Testing is a critical component of a robust MLOps workflow. It ensures:
-   **Data Quality**: Preventing bad data from corrupting downstream processes or models.
-   **Code Correctness**: Verifying that individual components and their interactions behave as expected.
-   **Model Performance**: Confirming that machine learning models meet performance benchmarks and generalize well.
-   **Reproducibility**: Ensuring that results can be consistently replicated.

All tests are located in the `tests/` directory and are executed using the `pytest` framework.

## 2. Types of Testing

### 2.1. Unit Testing

-   **Purpose**: To verify the correctness of individual functions, methods, or small modules in isolation.
-   **What is Checked**:
    -   Data preprocessing functions (e.g., in `src/gold_data_preprocessing/`, `src/silver_data_preprocessing/`).
    -   Utility functions (e.g., in `src/shared/utils.py`).
    -   Core logic components of the prediction server (e.g., `src/prediction_server/predict.py`).
    -   Custom transformer classes (e.g., `SimpleImputer`, `CategoricalEncoder`).
-   **Location**: `tests/test_data_preprocessing/`, `tests/test_data_ingestion/`, `tests/test_data_split/`, etc.
-   **How to Run**: `pytest tests/test_data_preprocessing/test_data_cleaning.py` (for a specific file) or `pytest tests/test_data_preprocessing/` (for a module).

### 2.2. Data Validation Testing (Great Expectations)

-   **Purpose**: To ensure the quality, integrity, and adherence to expected schemas and distributions of data at various stages of the data pipeline. This acts as a quality gate.
-   **What is Checked**:
    -   **Bronze Layer**: Basic raw data integrity (e.g., column presence, non-nullness, basic types). Located in `src/data_validation/great_expectations_bronze/`.
    -   **Silver Layer**: Cleaned and transformed data (e.g., no duplicates, correct feature engineering, consistent data types). Located in `src/data_validation/great_expectations_silver/`.
    -   **Gold Layer**: Model-ready data (e.g., final feature set, expected ranges after transformations, correct column order). Located in `src/data_validation/great_expectations_gold/`.
-   **How to Run**: Data validation checks are integrated directly into the data pipelines (`bronze_pipeline.py`, `silver_pipeline.py`, `gold_pipeline.py`). If validation fails, the data is quarantined, and a detailed report is generated.
-   **Reports**: HTML data docs are generated at `src/data_validation/great_expectations/gx/uncommitted/data_docs/local_site/index.html`.

### 2.3. Integration Testing

-   **Purpose**: To verify the correct interaction and data flow between multiple components or entire sub-systems.
-   **What is Checked**:
    -   End-to-end data pipeline stages (e.g., ensuring Bronze output correctly feeds into Silver input).
    -   Model training pipeline (e.g., data loading, preprocessing, and model training working together).
    -   Prediction server functionality (e.g., API endpoints correctly processing requests and returning predictions, including model loading and preprocessing within the server).
-   **Location**: `tests/test_pipelines/` and potentially specific tests within `tests/test_prediction_server/`.
-   **How to Run**: `pytest tests/test_pipelines/` or `pytest tests/test_prediction_server/`.

### 2.4. Model Validation Testing

-   **Purpose**: To assess the performance, stability, and generalization capabilities of the trained machine learning models.
-   **What is Checked**:
    -   **Performance Metrics**: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R² score on unseen validation and test sets.
    -   **Stability**: Consistency of performance across cross-validation folds.
    -   **Overfitting/Underfitting**: Analysis to ensure the model generalizes well and does not merely memorize training data.
    -   **Explainability Consistency**: Using SHAP to verify that the model's decision-making aligns with expectations and business logic.
-   **Location**: Results are logged to MLflow and extensively documented in the [Model Selection Report](../Modeling/model_selection_report.md) and [Champion Model Explainability Report](../Modeling/model_explainability_lgbm_champ.md).
-   **How to Run**: Executed as part of the `training_pipeline.py` and `tuning_pipeline.py`.

## 3. Running Tests

To run all tests in the project:

```bash
pytest
```

To run tests for a specific module or file:

```bash
pytest tests/test_data_preprocessing/
pytest tests/test_prediction_server/test_predict.py
```

It is recommended to run tests frequently during development to catch issues early.
