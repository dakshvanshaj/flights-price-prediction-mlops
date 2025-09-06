# Flight Price Prediction: An MLOps Project

Welcome to the documentation for this end-to-end flight price prediction project. This project demonstrates a complete MLOps workflow, from initial data exploration to model deployment and interpretation.

The primary goal is to predict flight prices accurately using a dataset of various flight details. This documentation serves as a central hub for all project artifacts, analysis, and reports.

---

## Project Journey

Follow the links below to explore the different stages of the project:

1.  **[Exploratory Data Analysis (EDA)](EDA/flights_eda.ipynb)**
    * Dive into the initial analysis of the dataset to understand its structure, identify patterns, and uncover key relationships between features.

2.  **[Model Selection Report](Modeling/model_selection_report.md)**
    * Discover how different models were evaluated and why **LightGBM** was chosen as the champion model based on its superior performance and efficiency.

3.  **[Champion Model Deep Dive (LightGBM)](Modeling/model_explainability_lgbm_champ.md)**
    * An in-depth look at the champion model using SHAP to understand its behavior both globally and on individual predictions.

4.  **[Champion vs. Challenger Analysis (LGBM vs. XGB)](Modeling/model_explainability_lgbm_vs_xgb.md)**
    * A comparative analysis to further validate our champion model and ensure its logic is sound and robust.

5.  **[Production Model Guide](LGBM_summary/LGBMR_production_model_details.md)**
    * Review the final details and characteristics of the production-ready LightGBM model.