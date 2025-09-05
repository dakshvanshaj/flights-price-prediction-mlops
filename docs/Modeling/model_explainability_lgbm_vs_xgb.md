# Model Comparison: LightGBM vs. XGBoost

## 1. Introduction

This document provides a comparative analysis of the champion model, **LightGBM**, and the primary challenger, **XGBoost**. The goal is to use SHAP (SHapley Additive exPlanations) to validate the choice of LightGBM by demonstrating its superior or more intuitive interpretability, and to provide further evidence that it is not overfitting but rather capturing true signals in the data.

## 2. Global Explainability Comparison

### A. SHAP Feature Importance (Bar Plot)

#### LightGBM

![SHAP Feature Importance LightGBM](../img/shap_lgbm_Feature%20Importance%20(Bar).png)

#### XGBoost

![SHAP Feature Importance XGBoost](../img/shap_xgbr_Feature%20Importance%20(Bar).png)

**Insights:**

*   Both models agree that **`time`** (flight duration) and **`flight_type`** are the two most important features.
*   The key difference lies in the third most important feature. LightGBM prioritizes **`from_location`**, while XGBoost gives more weight to **`agency`**.
*   This suggests that while both models capture similar high-level patterns, they have learned slightly different relationships in the data.

### B. SHAP Summary Plot

#### LightGBM

![SHAP Summary Plot LightGBM](../img/shap_lgbm_Summary%20Plot.png)

#### XGBoost

![SHAP Summary Plot XGBoost](../img/shap_xgbr_Summary%20Plot.png)

**Insights:**

*   The summary plots confirm the findings from the bar plots.
*   Both models show a clear and consistent positive correlation between `time` and the SHAP values, meaning longer flights lead to higher price predictions.
*   The categorical features like `flight_type` and `agency` show a wide distribution of SHAP values, indicating their strong predictive power.

## 3. Local Explainability Comparison

### A. SHAP Force Plot (XGBoost)

This interactive plot shows the forces driving individual predictions for the XGBoost model.

[View the interactive force plot](../img/shap_xgbr_force_plot.html)

### B. SHAP Waterfall Plots (XGBoost)

Here are the waterfall plots for the first three instances from the test set for the XGBoost model. The data for the instances used in the LightGBM plots can be found in the accompanying CSV file: [shap_local_instances.csv](../shap_local_instances.csv).

#### Instance 0

![Waterfall Plot for Instance 0 XGBoost](../img/shap_xgbr_Waterfall%20Plot%20for%20Instance%200.png)

#### Instance 1

![Waterfall Plot for Instance 1 XGBoost](../img/shap_xgbr_Waterfall%20Plot%20for%20Instance%201.png)

#### Instance 2

![Waterfall Plot for Instance 2 XGBoost](../img/shap_xgbr_Waterfall%20Plot%20for%20Instance%202.png)

**Insights:**

*   Similar to LightGBM, the XGBoost model's predictions are driven by a combination of features.
*   The local explanations for XGBoost also appear logical and consistent.

## 4. Feature Dependence Plot Comparison

### A. Time

#### LightGBM

![Dependence Plot - time LightGBM](../img/shap_lgbm_Dependence%20Plot%20-%20time.png)

#### XGBoost

![Dependence Plot - time XGBoost](../img/shap_xgbr_Dependence%20Plot%20-%20time.png)

**Insight:** Both models have learned a nearly identical positive linear relationship between `time` and its impact on the prediction.

### B. Flight Type

#### LightGBM

![Dependence Plot - flight_type LightGBM](../img/shap_lgbm_Dependence%20Plot%20-%20flight_type.png)

#### XGBoost

![Dependence Plot - flight_type XGBoost](../img/shap_xgbr_Dependence%20Plot%20-%20flight_type.png)

**Insight:** Both models show a similar categorical relationship for `flight_type`, with clear separation between the different classes.

### C. Agency

#### LightGBM

![Dependence Plot - agency LightGBM](../img/shap_lgbm_Dependence%20Plot%20-%20agency.png)

#### XGBoost

![Dependence Plot - agency XGBoost](../img/shap_xgbr_Dependence%20Plot%20-%20agency.png)

**Insight:** The `agency` feature shows a clearer separation of SHAP values in the XGBoost model compared to the LightGBM model. This suggests that XGBoost has learned a more discriminative relationship for this feature.

## 5. Conclusion: Reinforcing the Champion

The comparative SHAP analysis provides strong evidence to support the selection of LightGBM as the champion model:

1.  **Consistent Signal Detection:** Both models identified the same top-tier features (`time`, `flight_type`, `agency`, `from_location`), indicating that these are true, strong signals in the data, and that LightGBM is not hallucinating patterns.

2.  **Stability and Confidence:** The fact that two different powerful models learned similar relationships for the most important features gives us high confidence that our champion model is robust and not overfitting.

3.  **Performance is King:** While both models are interpretable and learn logical patterns, the **LightGBM model is both faster and more accurate**, as demonstrated in the *Model Selection Report*. The SHAP analysis confirms that this performance is not based on spurious correlations but on sound, explainable patterns.

This comparison solidifies our trust in the LightGBM model. It is not only the top performer but also a transparent and reliable model whose decisions can be understood and explained.
