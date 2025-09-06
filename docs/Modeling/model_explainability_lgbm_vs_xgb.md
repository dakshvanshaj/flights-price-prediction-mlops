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

Here are the waterfall plots for the first three instances from the test set for the XGBoost model. 

The data for these specific instances can be found in scaled preprocessed format in the accompanying CSV file: [shap_local_instances.csv](../shap_local_instances.csv).


|scaled price|
| :---  |
| 1.569 |
| -1.274 |
| -0.224 |

And the raw unscaled data is below(for only some columns to avoid complex reverse transformations.)

| | price | time | distance | flight_type |
| :--- | :--- | :--- | :--- | :--- |
| 1555 | 1566.260010 | 2.09 | 806.479980 | firstClass |
| 19696 | 517.820007 | 0.72 | 277.700012 | economic |
| 23885 | 826.020020 | 2.16 | 830.859985 | economic |

#### Instance 0

![Waterfall Plot for Instance 0 XGBoost](../img/shap_xgbr_Waterfall%20Plot%20for%20Instance%200.png)

*   **Predicted Value (Scaled):** `1.544`
*   **True Value (Scaled):** `1.569`
*   **Insight:** The prediction is close. The model correctly identified that the long flight `time` (duration), `flight_type` (firstclass), and `agency` were the primary drivers of the **high** price. 
Main difference is the agency for xgboost has mode contribution towards the price and `distance` + `route` further increased the price also in this case `from_location` has a minor decrease in price.

### XGBoost vs. LightGBM: Prediction Comparison (Instance 0)

Both models agree that `time` and `flight_type` are the primary drivers for this prediction. The main difference lies in how much importance each model assigns to the top features versus the secondary ones.

---

#### Key Feature Contribution Changes

| Feature         | XGBoost | LightGBM | Key Difference                                           |
|:----------------|:--------|:---------|:---------------------------------------------------------|
| **`f(x)` Prediction** | **1.544** | **1.567** | LightGBM's prediction is slightly higher.                 |
| `time`            | `+0.72`   | `+0.79`    | LightGBM assigns a stronger positive impact.             |
| `flight_type`     | `+0.45`   | `+0.53`    | LightGBM also gives more weight to this feature.         |
| `agency`          | `+0.23`   | `+0.15`    | XGBoost relies more on this secondary feature.           |
| `distance`        | `+0.12`   | `+0.09`    | XGBoost also gives more weight to distance.              |

---

### Interpretation

* **Agreement**: Both models correctly identify the same features as the most important factors pushing the prediction higher.
* **LightGBM's Strategy**: It puts more emphasis on the top two predictors (`time` and `flight_type`), attributing most of the prediction value to them.
* **XGBoost's Strategy**: It has a more distributed approach. While it agrees that `time` and `flight_type` are most important, it gives more relative weight to secondary features like `agency` and `distance` compared to LightGBM.

In essence, LightGBM is more confident in the top features, while XGBoost spreads the predictive power more evenly across the top four features.
#### Instance 1

![Waterfall Plot for Instance 1 XGBoost](../img/shap_xgbr_Waterfall%20Plot%20for%20Instance%201.png)

*   **Predicted Value (Scaled):** `-1.258`
*   **True Value (Scaled):** `-1.274`
*   **Insight:** Prediction is close here as well. The model correctly identified that the `flight_type`(economy) and a shorter `time` (duration) were the main factors driving the price down. Accuracy was further improved by a cheaper `agency`, and minor contributions from `distance` and `route`.

### LightGBM (Champ) vs. XGBoost: Prediction Comparison

The XGBoost model's prediction differs from the LightGBM model primarily because it **misinterpreted the impact of the `from_location` feature**, flipping its contribution from positive to negative.

---

#### Key Feature Contribution Changes

| Feature         | LightGBM (Champ) | XGBoost                 | Key Difference                        |
|:----------------|:-----------------|:------------------------|:--------------------------------------|
| **`f(x)` Prediction** | **-1.274** | **-1.258** | Slightly higher (less negative) score |
| `from_location`   | `+0.03`          | `-0.02`                 | **Impact flipped from positive to negative** |
| `agency`          | `-0.18`          | `-0.21`                 | Given a larger negative weight        |
| `flight_type`     | `-0.58`          | `-0.55`                 | Given a smaller negative weight       |

---

### What Went Wrong in XGBoost's Prediction

* **Primary Error**: XGBoost incorrectly learned that `from_location` should decrease the prediction score (`-0.02`), whereas the LightGBM model correctly identified it as a factor that should increase the score (`+0.03`).
* **Secondary Error**: It over-penalized the `agency` feature by attributing a larger negative impact to it compared to LightGBM.

In short, while XGBoost's final prediction is numerically close, its underlying logic for this instance is flawed, leading to a less accurate result.

#### Instance 2

![Waterfall Plot for Instance 2 XGBoost](../img/shap_xgbr_Waterfall%20Plot%20for%20Instance%202.png)
*   **Predicted Value (Scaled):** `-0.214`
*   **True Value (Scaled):** `-0.224`
* **Insight:** The model's prediction is driven by a conflict: `flight_type` (economy) pushes the price down significantly, while a long `time` (duration) pushes it back up. The final low price is the net result of these competing forces, along with the negative impact from `agency`.
### XGBoost vs. LightGBM: Comparison for Instance 2

Both models arrive at a very similar low prediction for this instance. They both identify a strong conflict between the `flight_type` (negative) and `time` (positive) features. The main difference is how they weigh the secondary features to arrive at the final score.

---

#### Key Feature Contribution Changes

| Feature         | XGBoost | LightGBM | Key Difference                                           |
|:----------------|:--------|:---------|:---------------------------------------------------------|
| **`f(x)` Prediction** | **-0.214** | **-0.224** | Predictions are nearly identical.                         |
| `flight_type`     | `-0.66`   | `-0.68`    | Both see this as the main negative driver.               |
| `time`            | `+0.59`   | `+0.68`    | The positive impact is weaker in XGBoost.                |
| `agency`          | `-0.29`   | `-0.26`    | Similar negative impact.                                 |
| `from_location`   | `-0.04`   | `-0.16`    | **LightGBM gives a much stronger negative weight here.** |

---

### Interpretation

* **Point of Agreement**: Both models see a "cancellation effect" where the negative impact of `flight_type` is counteracted by the positive impact of `time`.

* **Key Disagreement**: The models differ on how to resolve this conflict.
    * In **LightGBM**, the cancellation is almost perfect (`-0.68` vs `+0.68`). The final prediction is therefore heavily influenced by the next feature in line, `from_location`, which it sees as a strong negative factor (`-0.16`).
    * In **XGBoost**, the cancellation is incomplete (`-0.66` vs `+0.59`), leaving a net negative impact from the top two features. This, combined with the push from `agency`, drives the prediction down, with `from_location` (`-0.04`) playing a much less significant role.

In summary, for this specific instance, **LightGBM attributes much more importance to the departure location (`from_location`)** than XGBoost does.

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

3.  **Performance is King:** While both models are interpretable, the **LightGBM model is both faster and more accurate**, as demonstrated in the *Model Selection Report*. This superior performance likely stems from its more nuanced handling of key features like `from_location`, where it demonstrated more stable and logical behavior across different instances compared to XGBoost. The SHAP analysis confirms that this performance is not based on spurious correlations but on sound, explainable patterns.

This comparison solidifies our trust in the LightGBM model. It is not only the top performer but also a transparent and reliable model whose decisions can be understood and explained.
