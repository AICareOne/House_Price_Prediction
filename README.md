
# HDB Resale Price Prediction and Valuation

This repository contains the code and data used for predicting HDB resale prices and evaluating housing asset valuation. While HDB price prediction is one key part of this research, our work explores a broader multi-dimensional approach to evaluating housing assets, incorporating advanced feature engineering and machine learning methods to capture both numerical and categorical aspects—including location, property characteristics, and market seasonality.

## Data Source

The dataset is sourced from the [Data.gov.sg Resale Flat Prices Collection](https://data.gov.sg/collections/189/view). It contains detailed HDB resale transaction information such as:
- Transaction month
- Flat type
- Town
- Street name
- Floor area
- Storey range
- Lease commencement date
- Remaining lease (calculated when missing)

## Data Processing & Feature Engineering

Our data processing pipeline implements several techniques:

### Numerical and Categorical Preprocessing

Missing columns (e.g., `remaining_lease`) are filled by calculating the remaining lease as:

```
remaining_lease = (lease_commence_date + 99) - transaction_year
```

### Extended Location Features

In addition to target encoding categorical variables such as `town`, `flat_type`, `flat_model`, and `street_name`, we compute additional aggregated town statistics — mean, median, standard deviation, and count of the derived target (price per square meter).

### Seasonal & Time Features

The transaction month is converted into cyclical features (`month_sin`, `month_cos`) to capture seasonality, and the year is extracted.

### Correlation Analysis

A correlation study is conducted to evaluate the relationship between each feature and the resale price.

## Machine Learning Model & Parameter Tuning

We utilize an **XGBoost Regressor** as our core predictive model. Given a large dataset with over 272,000 records, the model is tuned with the following parameters:

- `n_estimators`: 300  
- `learning_rate`: 0.05  
- `max_depth`: 8  
- `subsample`: 0.8  
- `colsample_bytree`: 0.8  
- `random_state`: 42  
- `n_jobs`: -1  

### Model Evaluation

The model is evaluated using several metrics:

- **MAPE**: Approximately 5.64%  
- **MSE**: (Large numbers due to squaring errors, e.g., around 2.36 × 10⁹)  
- **RMSE**: Roughly \$48,586  
- **R²**: Approximately 0.94  

These results indicate that the model explains 94% of the variance in the HDB resale price, with relatively low average percentage error.

## Feature Importance and Model Explainability

To enhance model explainability, we use **SHAP (SHapley Additive exPlanations)**. SHAP values decompose a prediction into contributions from each feature, where:

- The **global expected value (baseline)** is the mean prediction over the training data.
- For an individual prediction, the sum of the baseline and all feature SHAP values equals the final prediction:

```
Prediction = Baseline + Σ SHAP_i
```

Because some features are correlated or overlapping (e.g., `town` and `street_name` both capture location), their individual contributions are distributed among them. You can interpret both the individual and grouped effects for further insight.

## How to Use the Code

### Training

1. **Data Preparation**  
   Place your CSV files (sourced from [Data.gov.sg](https://data.gov.sg/collections/189/view)) into the `Resale Transations/` folder.

2. **Run the Training Script**  
   The training code preprocesses the data, computes extended location features (including saving aggregated town statistics in `town_stats.joblib`), trains the XGBoost model with tuned hyperparameters, and saves the model and target encoder.

```bash
python model_train.py
```

---

## Deployment

1. **Deploy the Model**  
   The deployment code loads the saved model, target encoder, and town statistics.

2. **Make Predictions**  
   You can supply a new HDB record (even a subset of columns) as a Python dictionary or a Pandas DataFrame. The deployment script preprocesses the input, applies target encoding, and outputs:
   - The predicted resale price
   - The final feature set (after encoding)
   - The full processed input
   - The global expected (baseline) value and SHAP values showing each feature's contribution

```bash
python model_deploy.py
```

---

## Citation of Our Paper

If you use this repository in your research, please cite the following paper:

> **Zengxiang Li, Shen Ren, Nan Hu, Yong Liu, Zheng Qin, Rick Siow Mong Goh, Liwen Hou, Bharadwaj Veeravalli**  
> _Equality of Public Transit Connectivity: The Influence of MRT Services on Individual Buildings for Singapore_  
> **Transportmetrica B: Transport Dynamics**, 2018, IF 3.3  
> [DOI: 10.1080/21680566.2018.1449682](https://www.tandfonline.com/doi/full/10.1080/21680566.2018.1449682)

---

**Feel free to explore, comment, and contribute!**
