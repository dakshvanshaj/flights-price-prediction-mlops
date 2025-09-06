# Explaining the LightGBM Champion Model

## 1. Executive Summary

The LightGBM model was selected as the champion for its exceptional performance, achieving a CV RMSE of **$1.02** and a test RMSE of **$0.86**. This document delves into the model's decision-making process using SHAP (SHapley Additive exPlanations) to ensure its predictions are not only accurate but also transparent and interpretable.

The analysis confirms that the model has learned logical and robust patterns from the data. Key drivers of price predictions include **`time`**, **`flight_type`**, and **`from_location`**, which aligns with the feature importance identified during model evaluation. The model's behavior is consistent and its predictions can be trusted.

> **A Note on Interpreting Aggregated Features**
>
> In the SHAP plots throughout this document, you may notice features like `agency` and `route` displaying numerical values greater than 1 (e.g., `agency=2`). This is a cosmetic artifact of the visualization process and does not indicate an error in the model itself.
>
> This occurs because the explanation script aggregates multiple feature effects. For example, it groups the main `agency` feature with the engineered interaction feature `agency_flight_type`. The value `2` appears when both the base feature and the interaction feature are active for a given prediction.
>
> Therefore, the importance shown for a feature like `agency` represents the **combined impact of its main effect and its interactions**.
> However, the model importance for `agency_flight_type` and `route_agency` is near None, so it does not have an effect on the final conclusion. But it will be added to the fix list.
> The underlying SHAP calculations are sound, but this grouping should be kept in mind during interpretation.

## 2. Global Model Explainability

### A. SHAP Summary Plot

The summary plot provides a global overview of the model's feature importance and the impact of each feature on the predictions.

![SHAP Summary Plot](../img/shap_lgbm_Summary%20Plot.png)

**Insights:**

*   **`time`** is the most influential feature. Higher values of `time` (longer flight duration) have a strong positive impact on the predicted price.
*   **`flight_type`** is the second most important feature. It has a clear categorical impact, with different flight types pushing the prediction in opposite directions.
*   **`from_location`** and **`agency`** are also significant drivers.
*   The plot shows a clear separation of SHAP values for many features, indicating a strong predictive signal.

### B. SHAP Feature Importance (Bar Plot)

This plot shows the mean absolute SHAP value for each feature across all of the data. This quantifies, on average, the magnitude (positive or negative) of each feature's contribution towards the predicted prices. 
![SHAP Feature Importance](../img/shap_lgbm_Feature%20Importance%20(Bar).png)

**Insights:**

*   This plot confirms the findings from the summary plot, with **`time`**, **`flight_type`**, and **`from_location`** being the top three most important features.
*   This aligns with the feature importance from the LightGBM model itself, giving us confidence that the model is using these features as expected.

## 3. The Case of the Missing Temporal Features

An important observation from the EDA was that temporal features like `month` and `day_of_week` showed patterns related to flight prices. However, both the LightGBM and XGBoost models assigned them zero importance. This is not an error, but rather a common outcome in powerful tree-based models. Here's why this likely happened:

1.  **Feature Redundancy and Information Overlap:** The information contained in the temporal features was likely already captured by other, more powerful features. For instance, specific routes (`from_location` to `to_location`) or `agency` operations might have strong inherent seasonality. The model found that it could get more predictive power from these features, making the separate temporal features redundant.

2.  **The Power of High-Cardinality Features:** Features like `from_location` and `route` are very high-cardinality (they have many unique values). A single feature like `from_location` can implicitly learn seasonal effects (e.g., flights from a vacation destination are more expensive in the summer). The model found these features to be a more direct and powerful way to capture the variance in price than the more general `month` or `day_of_week` features.

3.  **Model's Focus on the Strongest Signals:** LightGBM and XGBoost are greedy algorithms that prioritize the features that provide the biggest predictive gains. The signals from `time` (duration), `flight_type`, and the route-related features were so overwhelmingly strong that the models could achieve very high accuracy by focusing on them alone. The marginal benefit of splitting on the weaker temporal features was so small that they were never chosen.

In essence, while the temporal features do have a relationship with price when viewed in isolation, their predictive information is more effectively captured by other features in the context of a multivariate model.

## 4. Local Model Explainability

### A. SHAP Force Plot

The force plot visualizes the SHAP values for a single prediction, showing how each feature contributes to pushing the prediction away from the base value.
For this interactive plot we placed multiple force plots for 2000 instances vertically and can visualize the patterns over the dataset.

[View the interactive force plot](../img/shap_lgbm_force_plot.html)

**Feature Combination Resulting In Low Price**
![Low Price](../img/forceplot_low_price.png)

**Insights:**
* We can see that a combination of lower time (flight duration) results in a significant decrease in price in combination with economy class a lower price is predicted which is intuitively sound.
* A costlier from_location in the particular selected instance pushed a price up little bit from the mean.

**Feature Combination Resulting In High Price**
![High Price](../img/forceplot_high_price1.png)

![High Price](../img/forceplot_high_price.png)

**Insights:**
* A higher flight duration(time) causes a massive increase in price.
* And when paired with Firstclass(Firstclass > Premium) of flights causes the highest price for flights.

**Feature Combination Resulting in Medium Price**
![Medium Price](../img/forceplot_Ltime_flightT2.png)
* This shows a combinational effect of low time duration but firstclass flightype including expensive from_location and route resulting in a medium level of price.
* Time being lower in this case is pulling the price down aggressively.

**Usage**
*   This interactive plot allows for the exploration of individual predictions.
*   Red bars represent features that increase the prediction, while blue bars represent features that decrease it.
*   The length of the bar indicates the magnitude of the feature's impact.
*   This is a powerful tool for understanding why the model made a specific prediction for a given flight.

### B. SHAP Waterfall Plots

Waterfall plots provide a detailed breakdown of a single prediction, showing how the SHAP values for each feature sum up to the final prediction.

These plots show how the model arrived at its final prediction for a single instance. The `f(x)` value at the top of the plot is the model's predicted output, and the `E[f(x)]` at the bottom is the base value (the average prediction over the entire dataset). Each bar in between shows how the value of each feature instance has pushed the prediction higher or lower.

The data for these specific instances can be found in scaled preprocessed format in the accompanying CSV file: [shap_local_instances.csv](../shap_local_instances.csv).


|scaled price|
| :---  |
| 1.568 |
| -1.273 |
| -0.223 |


And the raw unscaled data is below(for only some columns to avoid complex reverse transformations.)

| | price | time | distance | flight_type |
| :--- | :--- | :--- | :--- | :--- |
| 1555 | 1566.260010 | 2.09 | 806.479980 | firstClass |
| 19696 | 517.820007 | 0.72 | 277.700012 | economic |
| 23885 | 826.020020 | 2.16 | 830.859985 | economic |

> Check out these prediction comparison here for [lightgbm(champion) vs xgboost(challenger)](model_explainability_lgbm_vs_xgb.md) model to get even more clearer understanding on why lightgbm is performing better. 

#### Instance 0

![Waterfall Plot for Instance 0](../img/shap_lgbm_Waterfall%20Plot%20for%20Instance%200.png)

*   **Predicted Value (Scaled):** `1.567`
*   **True Value (Scaled):** `1.569`
*   **Insight:** The prediction is extremely accurate. The model correctly identified that the long flight `time` (duration) and `flight_type` (firstclass) were the primary drivers of the **high** price. Accuracy was increased from other minor factors; `agency` also had an additional positive impact on price increase while destination `to_location` has a minor decrease on price.

#### Instance 1

![Waterfall Plot for Instance 1](../img/shap_lgbm_Waterfall%20Plot%20for%20Instance%201.png)

*   **Predicted Value (Scaled):** `-1.274`
*   **True Value (Scaled):** `-1.274`
*   **Insight:** Another perfect prediction. The model correctly identified that the `flight_type`(economy) and a shorter `time` (duration) were the main factors driving the price down. Accuracy was further improved by a cheaper `agency`, and minor contributions from `distance` and `route`.

#### Instance 2

![Waterfall Plot for Instance 2](../img/shap_lgbm_Waterfall%20Plot%20for%20Instance%202.png)

*   **Predicted Value (Scaled):** `-0.224`
*   **True Value (Scaled):** `-0.224`
*   **Insight:** A perfect prediction. The model correctly identified that the `flight_type`(economy) was the main factor driving the price down, but this was counteracted by a longer `time` (duration), resulting in a lower price. However, here accuracy was further improved by deductions in price from `agency` and `from_location`, but an increase from `distance`. These all made it an accurate prediction.

## 5. Feature Dependence Plots

Dependence plots show how the SHAP value for a single feature changes as the feature's value changes.

### A. Time

![Dependence Plot - time](../img/shap_lgbm_Dependence%20Plot%20-%20time.png)

**Insight:** There is a clear positive linear relationship between `time` and its SHAP value. As the flight duration increases, the predicted price increases and due to `flighttype` there is some level of deviation for shap values for each flighttype.

### B. Flight Type

![Dependence Plot - flight_type](../img/shap_lgbm_Dependence%20Plot%20-%20flight_type.png)

**Insight:** This plot shows the categorical nature of `flight_type`. Different flight types have distinct SHAP values, indicating their different impacts on the price.

### C. Agency

![Dependence Plot - agency](../img/shap_lgbm_Dependence%20Plot%20-%20agency.png)

**Insight:** Similar to `flight_type`, `agency` shows a categorical relationship with the predicted price.

### D. Distance

![Dependence Plot - distance](../img/shap_lgbm_Dependence%20Plot%20-%20distance.png)

**Insight:** The relationship between `distance` and its SHAP value is mostly linear, with a slight curve. As distance increases, the impact on the price also increases. But time has much for well defined relation.

## 6. Conclusion

The SHAP analysis confirms that the LightGBM model is not a "black box". It has learned intuitive and explainable patterns from the data. The model's predictions are driven by logical features, and its behavior is consistent and trustworthy. This transparency is crucial for deploying the model in a production environment.
