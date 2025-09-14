# Flight Price Prediction: Model Selection Report

## 1. Executive Summary

This report details the systematic evaluation and iterative refinement of machine learning models for the flight price prediction task. The process involved establishing a baseline, tuning multiple advanced models, and conducting a final bake-off.

Initial results pointed to a LightGBM model with suspiciously high accuracy. This prompted a deep-dive investigation that uncovered and corrected a subtle overfitting issue caused by a leaky feature. After this refinement, a final, robust **LightGBM model was confirmed as the champion.**

The final model demonstrates an excellent balance of high performance and stability, with a **Cross-Validation RMSE of $9.57** and a **Final Test Set RMSE of $7.60**. This documentation tells the complete story, from a wide-ranging bake-off to the crucial investigative work that produced a truly reliable and production-ready model.

## 2. Initial Bake-Off & The "Too Good to Be True" Result

The first step was to compare our tuned tree-based models against a Linear Regression baseline.

| Model | CV R² Score | CV RMSE | CV RMSE Std Dev (Stability) | CV MAE | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LGBMRegressor (Tuned v1)** | **~1.000** | **$1.02** | **$0.38** | **$0.61** | **2.5 min** |
| RandomForestRegressor (Tuned) | 0.999 | $10.50 | $0.11 | $5.38 | 6.9 min |
| XGBoostRegressor (Tuned) | 0.999 | $11.95 | $3.32 | $9.48 | 1.9 min |
| LinearRegression (Base) | 0.986 | $42.64 | $0.18 | $34.32 | 2.2 min |

### Analysis and Red Flags

While the initial results were impressive across the board, the LightGBM model's performance was almost perfect (CV RMSE ~$1.02). Such high accuracy on a real-world dataset is a significant red flag for potential overfitting or data leakage. Furthermore, this initial model completely ignored temporal features, which contradicted our findings from the EDA. This warranted a deeper investigation.

## 3. Iteration 1: The Overfitting Trap

To simplify and optimize, a new iteration was run with two key changes:

1.  A new `is_tree_model` parameter was introduced to create a more efficient pipeline for tree-based models, bypassing unnecessary steps like one-hot encoding and scaling.
2.  Interaction features that showed zero importance in the initial SHAP analysis were removed.

This left a feature set that included `route` (a combination of origin and destination) and newly added cyclical temporal features.

### 3.1. Iteration 1: Results

| Model | CV RMSE | Final Model RMSE (Train+Val) | Overfitting Gap |
| :--- | :--- | :--- | :--- |
| **LightGBM** | **$7.25** | **$0.59** | **~92%** |
| **XGBoost** | **$6.47** | **$0.72** | **~89%** |

* The scores for LightGBM dropped to a reasonable range but the performance on the combined model is very overfitting for both.
* However, this time there was some level of importance given to `temporal` features which didnt happen before.
### 3.2. Diagnosis: Severe Overfitting

The results were clear: both models were **severely overfitting**. The error on the combined training and validation data (`Final Model RMSE`) was an order of magnitude lower than the average error during cross-validation (`CV RMSE`). This indicates the models were memorizing the training data and failing to generalize.

SHAP analysis of this run revealed that the engineered `route` feature had an overwhelmingly dominant contribution, dwarfing even `time` and `flight_type`. This pointed to `route` as the primary source of data leakage and overfitting.

## 4. Iteration 2: Taming the Model & Finding the True Champion

The clear next step was to remove the leaky `route` feature, forcing the models to learn from the more fundamental `from_location` and `to_location` features.

### 4.1. Iteration 2: Results

| Model | CV RMSE | Final Model RMSE (Train+Val) | Overfitting Gap | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **$9.57** | **$7.66** | **~20%** | **Stable & Reliable** |
| **XGBoost** | **$12.08** | **$0.90** | **~92%** | **Still Overfitting** |

### 4.2. Analysis and Final Decision

This iteration was the breakthrough:

1.  **LightGBM is the Champion:** By removing the `route` feature, the LightGBM model's performance stabilized. The CV RMSE and the final model RMSE are now closely aligned, indicating it generalizes well. The slight remaining gap is expected and healthy.
2.  **XGBoost is Dropped:** The XGBoost model, even without the `route` feature, continued to overfit severely. This made it an unreliable candidate for production.

**Conclusion: The stabilized LightGBM model from Iteration 2 was declared the definitive champion.**

## 5. Final Champion Model: Performance on Unseen Test Data

The final step was to evaluate the champion LightGBM model on the hold-out test set to confirm its real-world performance.

| Metric | Value |
| :--- | :--- |
| **R² Score** | **0.99956** |
| **Root Mean Squared Error (RMSE)** | **$7.60** |
| **Mean Absolute Error (MAE)** | **$5.50** |
| **Median Absolute Error** | **$4.21** |
| **Max Error** | **$39.94** |

The test set RMSE of **$7.60** is perfectly in line with the final model's training RMSE of **$7.66** and the cross-validation RMSE of **$9.57**. This consistency is the ultimate proof that the model is robust, reliable, and not overfit.

---

## Next Steps: Understanding the Champion

The metrics clearly show that our refined **LightGBM** is the champion model. The next stage of our analysis is to dive deep into its behavior to ensure it has learned logical and robust patterns from the data.

* **[Deep Dive into LightGBM's Behavior &raquo;](model_explainability_lgbm_champ.md)**
