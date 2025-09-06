# Data Pipelines

This document outlines the architecture and execution of the data pipelines used in this project. The pipelines are designed to process, validate, and transform data in stages, following a medallion architecture (Bronze, Silver, Gold) before it is used for model training and tuning.

## Overall Pipeline Architecture

The data flows through a series of pipelines, each responsible for a specific level of data quality and transformation.

```mermaid
graph TD
    A[Raw Data (.csv)] --> B{Bronze Pipeline};
    B -- Validation Pass --> C[Bronze Data (.csv)];
    B -- Validation Fail --> D[Quarantined Raw Data];
    C --> E{Silver Pipeline};
    E -- Validation Pass --> F[Silver Data (.parquet)];
    E -- Validation Fail --> G[Quarantined Bronze Data];
    F --> H{Gold Pipeline};
    H -- Validation Pass --> I[Gold Data (.parquet)];
    H -- Validation Fail --> J[Quarantined Silver Data];
    I --> K{Training Pipeline};
    I --> L{Tuning Pipeline};
    K -- Logs --> M[MLflow Tracking];
    L -- Logs --> M;
    K -- Produces --> N[Trained Model];
    L -- Produces --> O[Best Hyperparameters];
```

---

## 1. Bronze Pipeline

The Bronze pipeline is the first entry point for raw data into the system. Its primary responsibility is to act as an initial quality gate, ensuring that incoming data conforms to a basic, expected schema and structure.

-   **Source Code:** `src/pipelines/bronze_pipeline.py`

### Purpose

-   To validate the structure and basic quality of raw data files using Great Expectations.
-   To separate valid data from invalid data, preventing "garbage in, garbage out."

### Key Steps

1.  **Initialize Great Expectations (GE) Context**: Sets up the GE environment.
2.  **Define Data Source and Asset**: Points GE to the raw data directory and specifies how to read the CSV files.
3.  **Build and Apply Expectation Suite**: Uses the `build_bronze_expectations` suite to check for things like column presence, non-nullness, and basic type adherence.
4.  **Run Checkpoint**: Executes the validation.
5.  **Move File Based on Result**:
    -   **On Success**: Moves the raw file to the `data/bronze_data/processed/` directory.
    -   **On Failure**: Moves the raw file to the `data/bronze_data/quarantined/` directory.

### How to Run

Execute the script from the root directory, providing the name of the raw data file.

```bash
python src/pipelines/bronze_pipeline.py <file_name.csv>
```

**Example:**

```bash
python src/pipelines/bronze_pipeline.py train.csv
```

---

## 2. Silver Pipeline

The Silver pipeline takes the validated data from the Bronze layer and begins the process of cleaning, standardizing, and enriching it.

-   **Source Code:** `src/pipelines/silver_pipeline.py`

### Purpose

-   To clean and standardize data.
-   To perform initial feature engineering, such as extracting features from dates.
-   To enforce a consistent schema and data types.

### Key Steps

1.  **Data Ingestion**: Loads a file from the Bronze processed directory.
2.  **Preprocessing & Cleaning**:
    -   Renames columns for clarity.
    -   Standardizes column names to a consistent format (e.g., snake_case).
    -   Optimizes data types (e.g., converting strings to numeric/datetime).
    -   Sorts data by date to prepare for time-series analysis.
    -   Handles erroneous duplicates.
3.  **Feature Engineering**: Creates new features from existing ones (e.g., `day`, `month`, `year` from a `date` column).
4.  **Enforce Schema**: Reorders columns to a predefined, consistent order.
5.  **Data Validation**: Runs a `silver_expectations` suite with Great Expectations to ensure the output data meets higher quality standards (e.g., correct data types, no nulls in critical columns).
6.  **Save Data**:
    -   **On Success**: Saves the processed DataFrame as a Parquet file to `data/silver_data/processed/`.
    -   **On Failure**: Saves the failed DataFrame to `data/silver_data/quarantined/`.

### How to Run

Execute the script with the name of the file from the Bronze processed directory.

```bash
python src/pipelines/silver_pipeline.py <bronze_file_name.csv>
```

**Example:**

```bash
python src/pipelines/silver_pipeline.py train.csv
```

---

## 3. Gold Pipeline

The Gold pipeline is the final and most intensive transformation stage. It prepares the data specifically for machine learning by applying complex feature engineering and preprocessing steps.

-   **Source Code:** `src/pipelines/gold_pipeline.py`

### Purpose

-   To create a feature-rich, analysis-ready dataset for modeling.
-   To handle missing values, encode categorical variables, scale numerical features, and manage outliers.
-   To save the fitted preprocessing objects (like scalers and encoders) from the training run so they can be applied to validation and test data consistently.

### Key Steps

The pipeline is executed differently for training data versus validation/test data.

**On Training Data:**
1.  **Data Ingestion**: Loads data from the Silver layer.
2.  **Data Cleaning**: Drops unnecessary columns and duplicates.
3.  **Imputation**: Fits an imputer on the training data to learn strategies for filling missing values and then transforms the data.
4.  **Feature Engineering**: Creates cyclical and interaction features.
5.  **Rare Category Grouping**: Groups infrequent categorical values into a single "rare" category.
6.  **Categorical Encoding**: Fits an encoder and transforms categorical columns into a numerical format.
7.  **Outlier Handling**: Detects and mitigates the effect of outliers.
8.  **Power Transformations**: Applies transformations (e.g., Yeo-Johnson) to make data distributions more Gaussian-like.
9.  **Scaling**: Fits a scaler (e.g., StandardScaler) and scales numerical features.
10. **Final Validation**: Runs a `gold_expectations` suite to ensure the final data is ready for modeling.
11. **Save Data & Objects**: Saves the processed Gold data and serializes all the fitted preprocessing objects (imputer, encoder, scaler, etc.) to disk.

**On Validation/Test Data:**
- The pipeline loads the already-fitted preprocessing objects and uses them to `transform` the new data. This prevents data leakage and ensures consistent transformations.

### How to Run

The main function in the script orchestrates the processing for the `train`, `validation`, and `test` splits automatically.

```bash
python src/pipelines/gold_pipeline.py
```

---

## 4. Training & Tuning Pipelines

Once the Gold data is ready, it can be used for model training and hyperparameter tuning.

### Training Pipeline

-   **Source Code:** `src/pipelines/training_pipeline.py`
-   **Purpose**: To train a model, evaluate its performance, and log all relevant artifacts (metrics, parameters, plots, and the model itself) to MLflow.
-   **Features**:
    -   Supports both simple train/validation splits and time-based cross-validation.
    -   Logs detailed evaluation metrics for both scaled and unscaled predictions.
    -   Generates and logs interpretability artifacts like feature importance and SHAP plots.
    -   Can register the trained model in the MLflow Model Registry.
-   **How to Run**:
    ```bash
    python src/pipelines/training_pipeline.py <train_file.parquet> <validation_file.parquet> --test_file_name <test_file.parquet>
    ```

### Tuning Pipeline

-   **Source Code:** `src/pipelines/tuning_pipeline.py`
-   **Purpose**: To systematically find the best hyperparameters for a model.
-   **Features**:
    -   Integrates with `tuning.yaml` for flexible configuration.
    -   Supports multiple tuning strategies: Grid Search, Random Search, Halving Search, and Optuna.
    -   Logs all trials and the best results to MLflow.
-   **How to Run**:
    ```bash
    python src/pipelines/tuning_pipeline.py <train_file.parquet>
    ```
