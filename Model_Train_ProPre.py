#!/usr/bin/env python3
"""
Model Training Script for HDB House Price Prediction

Features:
1. Data Loading from multiple CSV files.
2. Extended Location Features: For "town", beyond target encoding, aggregated statistics (mean, median, std, count)
   are computed on the derived target (price per square meter) from the training set and merged back.
3. Inclusion of fine-grained location (street_name) using categorical (target) encoding.
4. Feature engineering for time-related features.
5. Improved XGBoost hyperparameters for a larger training set.
6. Point prediction model and probabilistic forecasts via quantile regression.
7. Saving of trained models and artifacts for deployment.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib

# Set backend for plotting; you can try 'TkAgg' or 'Qt5Agg' as needed.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance, DMatrix, train as xgb_train
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib  # For saving and loading the model
import category_encoders as ce  # For target encoding

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
csv_files = glob.glob("Resale Transations/*.csv")
if not csv_files:
    raise ValueError("No CSV files found in the specified folder.")
print("Loading CSV files:", csv_files)

df_list = []
for file in csv_files:
    df_tmp = pd.read_csv(file)
    # If the file is missing 'remaining_lease', compute it if possible.
    if 'remaining_lease' not in df_tmp.columns:
        if 'lease_commence_date' in df_tmp.columns and 'month' in df_tmp.columns:
            df_tmp['lease_commence_date'] = pd.to_numeric(df_tmp['lease_commence_date'], errors='coerce')
            df_tmp['month'] = pd.to_datetime(df_tmp['month'], format='%Y-%m', errors='coerce')
            df_tmp['remaining_lease'] = (df_tmp['lease_commence_date'] + 99) - df_tmp['month'].dt.year
        else:
            df_tmp['remaining_lease'] = np.nan
    df_list.append(df_tmp)

df = pd.concat(df_list, ignore_index=True)

print("Unique values in 'month' column:")
print(df['month'].unique())
df['month'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
print("Date range in dataset:", df['month'].min(), "to", df['month'].max())
df = df.dropna(subset=['month'])

# Retain "street_name" because we want fine-grained location info.
if 'block' in df.columns:
    df.drop(columns=['block'], inplace=True)


# -------------------------------
# Preserve Floor Range and Other Important Factors
# -------------------------------
def convert_storey_range(s):
    try:
        lower, upper = s.split(" to ")
        return (int(lower) + int(upper)) / 2
    except Exception:
        return np.nan


if 'storey_range' in df.columns:
    df['storey_range_numeric'] = df['storey_range'].apply(convert_storey_range)
else:
    raise ValueError("The dataset must contain a 'storey_range' column.")

for col in ['floor_area_sqm', 'lease_commence_date', 'remaining_lease']:
    if col not in df.columns:
        raise ValueError(f"Important column '{col}' is missing from the dataset.")

# -------------------------------
# Derived Target: Price per Square Meter
# -------------------------------
df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

# -------------------------------
# 2. Time-Based Split
# -------------------------------
cutoff_date = pd.Timestamp("2024-04-01")
train_df = df[df['month'] < cutoff_date].copy()
test_df = df[df['month'] >= cutoff_date].copy()
print(f"\nTraining set size: {train_df.shape}")
print(f"Testing set size: {test_df.shape}")
if train_df.empty or test_df.empty:
    raise ValueError("One of the time-based splits is empty. Please adjust your cutoff date.")

# -------------------------------
# 2.5 Add Extended 'Town' Features (Aggregated Statistics)
# -------------------------------
town_stats_df = train_df.groupby("town")["price_per_sqm"].agg(['mean', 'median', 'std', 'count']).reset_index()
town_stats_df.rename(columns={
    'mean': 'town_mean',
    'median': 'town_median',
    'std': 'town_std',
    'count': 'town_count'
}, inplace=True)

town_stats_dict = {}
for _, row in town_stats_df.iterrows():
    town_key = str(row['town']).upper()
    town_stats_dict[town_key] = {
        'town_mean': row['town_mean'],
        'town_median': row['town_median'],
        'town_std': row['town_std'],
        'town_count': row['town_count']
    }
joblib.dump(town_stats_dict, "town_stats.joblib")
print("Town aggregated statistics saved to town_stats.joblib")
train_df = pd.merge(train_df, town_stats_df, on='town', how='left')
test_df = pd.merge(test_df, town_stats_df, on='town', how='left')

# -------------------------------
# 3. Define Features and Target
# -------------------------------
target_variable = 'resale_price'
if target_variable not in df.columns:
    raise ValueError(f"Target column '{target_variable}' not found.")

# Use:
# - For location: original "town" (target encoded) + extended features, and "street_name"
# - Other categorical: 'flat_type', 'flat_model'
# - Numeric: 'storey_range_numeric', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease'
# - Time-based: 'year', 'month_sin', 'month_cos'
X_train = train_df.copy()
X_test = test_df.copy()
X_test_output = X_test.copy()

# -------------------------------
# 4. Engineer Time-Based Features
# -------------------------------
X_train['year'] = X_train['month'].dt.year
X_train['month_num'] = X_train['month'].dt.month
X_test['year'] = X_test['month'].dt.year
X_test['month_num'] = X_test['month'].dt.month

X_train['month_sin'] = np.sin(2 * np.pi * X_train['month_num'] / 12)
X_train['month_cos'] = np.cos(2 * np.pi * X_train['month_num'] / 12)
X_test['month_sin'] = np.sin(2 * np.pi * X_test['month_num'] / 12)
X_test['month_cos'] = np.cos(2 * np.pi * X_test['month_num'] / 12)

X_train.drop(columns=['month', 'month_num'], inplace=True)
X_test.drop(columns=['month', 'month_num'], inplace=True)

# -------------------------------
# Convert Lease-related Columns to Numeric
# -------------------------------
X_train['lease_commence_date'] = pd.to_numeric(X_train['lease_commence_date'], errors='coerce')
X_test['lease_commence_date'] = pd.to_numeric(X_test['lease_commence_date'], errors='coerce')


def convert_remaining_lease(s):
    try:
        parts = s.split()
        years = int(parts[0])
        months = int(parts[2])
        return years + months / 12.0
    except Exception:
        return np.nan


X_train['remaining_lease'] = X_train['remaining_lease'].apply(
    lambda x: convert_remaining_lease(x) if isinstance(x, str) else x)
X_test['remaining_lease'] = X_test['remaining_lease'].apply(
    lambda x: convert_remaining_lease(x) if isinstance(x, str) else x)

# -------------------------------
# 5. Categorical Encoding
# -------------------------------
cat_cols = ['town', 'flat_type', 'flat_model', 'street_name']
for col in cat_cols:
    X_train[col] = X_train[col].fillna("unknown")
    X_test[col] = X_test[col].fillna("unknown")

encoder = ce.TargetEncoder(cols=cat_cols)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], X_train['price_per_sqm'])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# -------------------------------
# 6. Final Feature Selection for Modeling
# -------------------------------
features = [
    'town', 'flat_type', 'flat_model', 'street_name',
    'town_mean', 'town_median', 'town_std', 'town_count',
    'storey_range_numeric', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease',
    'year', 'month_sin', 'month_cos'
]
X_train_model = X_train[features]
X_test_model = X_test[features]

y_train = X_train[target_variable]
y_test = X_test[target_variable]

print("\nFeatures used for modeling:")
print(X_train_model.columns.tolist())

# -------------------------------
# 7. Correlation Analysis on Training Data
# -------------------------------
train_corr = X_train_model.copy()
train_corr[target_variable] = y_train
correlations = train_corr.corr()[target_variable].drop(target_variable).sort_values(ascending=False)
print("\nFeature Correlations with House Price (resale_price) on Training Data:")
print(correlations)
correlations.to_csv("feature_correlations.csv", header=['Correlation'])

# -------------------------------
# 8. Model Training with Improved Parameters (Point Model)
# -------------------------------
point_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
point_model.fit(X_train_model, y_train)
print("\nXGBoost point prediction model training complete.")

y_pred = point_model.predict(X_test_model)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nPoint Model Evaluation Metrics:")
print("MAPE: {:.2f}%".format(mape))
print("MSE: {:.2f}".format(mse))
print("RMSE: {:.2f}".format(rmse))
print("R²: {:.2f}".format(r2))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Actual vs Predicted Resale Prices")
plt.xlabel("Actual Resale Price")
plt.ylabel("Predicted Resale Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plot_importance(point_model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()

# -------------------------------
# 9. Probabilistic Forecasting: Quantile Regression with XGBoost
# -------------------------------
from xgboost import DMatrix, train as xgb_train


def xgb_quantile_obj(q):
    """
    Custom quantile objective (pinball loss) for XGBoost.
    """

    def objective(y_pred, dtrain):
        y_true = dtrain.get_label()
        grad = np.where(y_true > y_pred, -q, 1 - q)
        hess = np.ones_like(grad)
        return grad, hess

    return objective


# Scale targets for quantile regression.
scale_factor = 1e5
y_train_scaled = y_train / scale_factor
y_test_scaled = y_test / scale_factor

dtrain = DMatrix(X_train_model, label=y_train_scaled)
dtest = DMatrix(X_test_model, label=y_test_scaled)

quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
quantile_models = {}
quantile_predictions_scaled = {}

xgb_params = {
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0
}

print("\nTraining quantile regression models:")
for q in quantiles:
    print(f"Training for quantile: {q}")
    model_q = xgb_train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=250,
        obj=xgb_quantile_obj(q)
    )
    quantile_models[q] = model_q
    model_q.save_model(f"xgb_model_q{int(q * 100)}.json")
    quantile_predictions_scaled[q] = model_q.predict(dtest)

# Re-scale quantile predictions back to original scale.
quantile_predictions = {q: preds * scale_factor for q, preds in quantile_predictions_scaled.items()}
joblib.dump(quantile_models, "xgb_quantile_models.joblib")
print("\nAll quantile models saved.")


def pinball_loss(y_true, y_pred, q):
    err = y_true - y_pred
    return np.mean(np.maximum(q * err, (q - 1) * err))


print("\nQuantile Model Evaluation (Pinball Loss):")
for q in quantiles:
    loss = pinball_loss(y_test, quantile_predictions[q], q)
    print(f"Quantile {q}: Pinball Loss = {loss:.2f}")

# -------------------------------
# 10. Output Processed Validation Data with Predictions
# -------------------------------
validation_results = X_test_model.copy()
validation_results['predicted_resale_price'] = y_pred
validation_results['actual_resale_price'] = y_test.values

for q in quantiles:
    col_name = f"predicted_resale_price_q{int(q * 100)}"
    validation_results[col_name] = quantile_predictions[q]

validation_results['month'] = X_test_output['month'].dt.strftime('%Y-%m')
validation_results['storey_range'] = X_test_output['storey_range']
validation_results['floor_area_sqm'] = X_test_output['floor_area_sqm']
validation_results['lease_commence_date'] = X_test_output['lease_commence_date']
validation_results['remaining_lease'] = X_test_output['remaining_lease']

cols_order = [
                 'month', 'storey_range',
                 'town', 'flat_type', 'flat_model', 'street_name',
                 'town_mean', 'town_median', 'town_std', 'town_count',
                 'storey_range_numeric', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease',
                 'year', 'month_sin', 'month_cos',
                 'actual_resale_price', 'predicted_resale_price'
             ] + [f"predicted_resale_price_q{int(q * 100)}" for q in quantiles]

validation_results = validation_results[cols_order]
output_csv = "validation_predictions_quantile.csv"
validation_results.to_csv(output_csv, index=False)
print(f"\nValidation dataset with quantile predictions saved to: {output_csv}")

# -------------------------------
# 11. Save Artifacts for Deployment
# -------------------------------
point_model_file = "xgboost_house_price_model_propre.joblib"
joblib.dump(point_model, point_model_file)
print("\nPoint model saved to:", point_model_file)

encoder_file = "target_encoder.joblib"
joblib.dump(encoder, encoder_file)
print("Target encoder saved to:", encoder_file)

# -------------------------------
# 12. (Optional) Model Deployment Example
# -------------------------------
loaded_point_model = joblib.load(point_model_file)
print("Point model loaded for prediction.")

new_data = X_test_model.iloc[0:1]
predicted_price = loaded_point_model.predict(new_data)
print("\nPrediction for a new data point (point model):")
print("Input features:\n", new_data)
print("Predicted resale price (point model): {:.2f}".format(predicted_price[0]))

loaded_quantile_models = joblib.load("xgb_quantile_models.joblib")
dnew = DMatrix(new_data)
print("\nPrediction for a new data point (quantile models):")
for q in quantiles:
    pred_q = loaded_quantile_models[q].predict(dnew)
    print(f"Quantile {q}: {pred_q[0] * scale_factor:.2f}")


# -------------------------------
# 13. Visualization of Probabilistic Prediction Range
# -------------------------------
def plot_quantile_fan(quantile_preds, point_prediction):
    """
    Plot the quantile predictions as a fan chart.
    - quantile_preds: Dictionary mapping quantile to scalar prediction (in original scale) for the instance.
    - point_prediction: Point prediction for the instance (in original scale).
    """
    quantiles_sorted = sorted(list(quantile_preds.keys()))
    pred_values = [quantile_preds[q][0] if isinstance(quantile_preds[q], np.ndarray) else quantile_preds[q] for q in
                   quantiles_sorted]

    plt.figure(figsize=(8, 6))
    plt.plot(quantiles_sorted, pred_values, marker='o', label='Quantile Predictions')
    plt.hlines(point_prediction[0], quantiles_sorted[0], quantiles_sorted[-1], colors='red', linestyles='--',
               label='Point Prediction')
    plt.fill_between(quantiles_sorted, pred_values, point_prediction[0], color='gray', alpha=0.3,
                     label='Prediction Interval')
    plt.xlabel("Quantile")
    plt.ylabel("Predicted Resale Price")
    plt.title("Probabilistic Forecasting: Quantile Predictions Fan Chart")
    plt.legend()
    plt.grid(True)
    plt.show()


print("\nVisualizing probabilistic prediction range (fan chart):")
plot_quantile_fan(quantile_predictions, y_pred.reshape(-1, 1))
