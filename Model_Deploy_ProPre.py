#!/usr/bin/env python3
"""
Model Deployment Script (Probabilistic Forecasting):
 - Loads the saved XGBoost point model, quantile models, target encoder, and town aggregated statistics.
 - Accepts an input (dictionary or DataFrame) that may contain a subset of the table columns.
 - Handles missing categorical values by filling them with "unknown".
 - Preprocesses the input (date conversion, time-based features, storey_range conversion, lease conversion).
 - Fills in extended town aggregated features (town_mean, town_median, town_std, town_count) from saved statistics.
 - Prepares the final features and applies target encoding.
 - Casts all features to numeric types so that XGBoost does not complain.
 - Predicts the house price using the point model and computes SHAP values for explainability.
 - Also computes probabilistic predictions (quantiles) using the quantile models.
 - Visualizes the quantile prediction fan (i.e. prediction interval) alongside the point prediction.
"""

import pandas as pd
import numpy as np
import joblib
import category_encoders as ce
from xgboost import XGBRegressor, DMatrix
import shap

import matplotlib

# Set backend for plotting; you can try 'TkAgg' or 'Qt5Agg' as needed.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Configure pandas to display all columns.
pd.set_option('display.max_columns', None)


def load_artifacts():
    """
    Load and return the saved point model, target encoder, town aggregated statistics, and quantile models.
    Assumes that "xgboost_house_price_model_propre.joblib", "target_encoder.joblib", and "town_stats.joblib"
    exist; quantile models are loaded from "xgb_quantile_models.joblib".
    """
    point_model = joblib.load("xgboost_house_price_model_propre.joblib")
    encoder = joblib.load("target_encoder.joblib")
    try:
        town_stats = joblib.load("town_stats.joblib")
    except Exception:
        town_stats = {}
    try:
        quantile_models = joblib.load("xgb_quantile_models.joblib")
    except Exception:
        quantile_models = None
    return point_model, encoder, town_stats, quantile_models


def convert_storey_range(s):
    """
    Convert a storey_range string (e.g., "04 to 06") to a numeric value (midpoint).
    """
    try:
        lower, upper = s.split(" to ")
        return (int(lower) + int(upper)) / 2
    except Exception:
        return np.nan


def convert_remaining_lease(s):
    """
    Convert a remaining_lease string (e.g., "60 years 08 months") to a numeric value in years.
    """
    try:
        parts = s.split()
        years = int(parts[0])
        months = int(parts[2])
        return years + months / 12.0
    except Exception:
        return np.nan


def preprocess_input(df):
    """
    Preprocess the input DataFrame:
      - Ensure required columns exist (if missing, add defaults).
      - Fill missing values for expected columns.
      - Convert 'month' (format "YYYY-MM") to datetime.
      - Create time-based features: 'year', 'month_sin', 'month_cos'.
      - Convert 'storey_range' into 'storey_range_numeric'.
      - Convert lease-related columns to numeric.
    """
    expected_defaults = {
        "month": np.nan,              # Expected as a string "YYYY-MM"
        "storey_range": np.nan,       # e.g., "04 to 06"
        "town": "unknown",
        "flat_type": "unknown",
        "flat_model": "unknown",
        "street_name": "unknown",     # Fine-grained location info.
        "floor_area_sqm": np.nan,
        "lease_commence_date": np.nan,
        "remaining_lease": np.nan
    }
    for col, default in expected_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    if df['month'].notna().any():
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
    else:
        df['month'] = pd.NaT

    df['year'] = df['month'].dt.year
    df['month_num'] = df['month'].dt.month
    df['month_sin'] = df['month_num'].apply(lambda m: np.sin(2 * np.pi * m / 12) if pd.notnull(m) else np.nan)
    df['month_cos'] = df['month_num'].apply(lambda m: np.cos(2 * np.pi * m / 12) if pd.notnull(m) else np.nan)
    df.drop(columns=['month_num'], inplace=True)

    df['storey_range_numeric'] = df['storey_range'].apply(lambda x: convert_storey_range(x) if pd.notnull(x) else np.nan)
    df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
    df['remaining_lease'] = df['remaining_lease'].apply(lambda x: convert_remaining_lease(x) if isinstance(x, str) else x)

    return df


def fill_town_aggregates(df, town_stats):
    """
    Fill in extended town features (town_mean, town_median, town_std, town_count) based on town_stats.
    """
    def get_stats(town):
        key = str(town).upper()
        stats = town_stats.get(key, None)
        if stats is None:
            return pd.Series([np.nan, np.nan, np.nan, np.nan],
                             index=['town_mean', 'town_median', 'town_std', 'town_count'])
        else:
            return pd.Series([
                stats.get("town_mean", np.nan),
                stats.get("town_median", np.nan),
                stats.get("town_std", np.nan),
                stats.get("town_count", np.nan)
            ], index=['town_mean', 'town_median', 'town_std', 'town_count'])
    if 'town' in df.columns:
        stats_df = df['town'].apply(lambda x: get_stats(x if pd.notnull(x) else "unknown"))
        df = pd.concat([df, stats_df], axis=1)
    else:
        df['town_mean'] = np.nan
        df['town_median'] = np.nan
        df['town_std'] = np.nan
        df['town_count'] = np.nan
    return df


def prepare_features(df, town_stats):
    """
    Prepare the final feature set.
    Expected features:
      - Categorical (to be target encoded): 'town', 'flat_type', 'flat_model', 'street_name'
      - Extended town features: 'town_mean', 'town_median', 'town_std', 'town_count'
      - Numeric: 'storey_range_numeric', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease'
      - Time-based: 'year', 'month_sin', 'month_cos'
    """
    df = fill_town_aggregates(df, town_stats)
    required_features = [
        'town', 'flat_type', 'flat_model', 'street_name',
        'town_mean', 'town_median', 'town_std', 'town_count',
        'storey_range_numeric', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease',
        'year', 'month_sin', 'month_cos'
    ]
    for col in required_features:
        if col not in df.columns:
            df[col] = np.nan
    return df[required_features]


def predict_house_price(input_data):
    """
    Predict point resale price using the point model.
    Returns:
      (predicted_price, X_input, processed_input)
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("Input data must be a dictionary or a DataFrame.")

    processed_input = preprocess_input(df_input)
    _, _, town_stats, _ = load_artifacts()
    X_input = prepare_features(processed_input, town_stats)
    X_input = X_input.copy()
    categorical_cols = ['town', 'flat_type', 'flat_model', 'street_name']
    for col in categorical_cols:
        X_input.loc[:, col] = X_input[col].fillna("unknown")
    point_model, encoder, _, _ = load_artifacts()
    X_input.loc[:, categorical_cols] = encoder.transform(X_input[categorical_cols]).astype(float)
    X_input = X_input.astype(float)
    predicted_price = point_model.predict(X_input)
    return predicted_price, X_input, processed_input


def predict_quantiles(X_input, quantile_models, scale_factor=1e5):
    """
    Predict quantile values using the quantile models.
    Returns a dictionary mapping quantile to prediction (rescaled to original scale).
    """
    from xgboost import DMatrix
    dinput = DMatrix(X_input)
    quantile_preds = {}
    for q, model_q in quantile_models.items():
        preds_scaled = model_q.predict(dinput)
        quantile_preds[q] = preds_scaled * scale_factor
    return quantile_preds


def compute_shap_explanation(model, X_input):
    """
    Compute SHAP values for the point model.
    Returns (global_expected_value, shap_df)
    """
    explainer = shap.TreeExplainer(model)
    global_expected_value = explainer.expected_value
    shap_values = explainer.shap_values(X_input)
    shap_df = pd.DataFrame({
        'Feature': X_input.columns,
        'SHAP Value': shap_values[0]
    })
    return global_expected_value, shap_df


def plot_quantile_fan(quantile_preds, point_prediction):
    """
    Plot the quantile predictions as a fan chart.
    quantile_preds: dictionary mapping quantile to scalar prediction (original scale).
    point_prediction: array with the point prediction.
    """
    quantiles_sorted = sorted(list(quantile_preds.keys()))
    pred_values = [quantile_preds[q][0] if isinstance(quantile_preds[q], np.ndarray) else quantile_preds[q] for q in quantiles_sorted]

    plt.figure(figsize=(8,6))
    plt.plot(quantiles_sorted, pred_values, marker='o', label='Quantile Predictions')
    plt.hlines(point_prediction[0], quantiles_sorted[0], quantiles_sorted[-1], colors='red', linestyles='--', label='Point Prediction')
    plt.fill_between(quantiles_sorted, pred_values, point_prediction[0], color='gray', alpha=0.3, label='Prediction Interval')
    plt.xlabel("Quantile")
    plt.ylabel("Predicted Resale Price")
    plt.title("Probabilistic Forecasting: Quantile Prediction Fan Chart")
    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    # Sample input (can be dictionary or DataFrame).
    sample_input = {
        "month": "2025-01",
        "storey_range": "04 to 06",
        "town": "YISHUN",
        "flat_type": "4-room",
        "flat_model": "Simplified",
        "street_name": "ANG MO KIO AVE 10",
        "floor_area_sqm": 84,
        "lease_commence_date": "1985",
        "remaining_lease": "59 years 11 months"
    }

    # Predict point resale price.
    point_pred, features_used, processed = predict_house_price(sample_input)
    print("Predicted House Price (Point Model):", point_pred)
    print("\nFinal Features used for point prediction:")
    print(features_used.to_string(index=False))
    print("\nProcessed Input Data:")
    print(processed.to_string(index=False))

    # Compute SHAP explanation for the point model.
    point_model, encoder, town_stats, quantile_models = load_artifacts()
    global_expected_value, shap_df = compute_shap_explanation(point_model, features_used)
    print("\nGlobal Expected Value (Baseline):", global_expected_value)
    print("\nInput Variable Importance (SHAP values) for the point prediction:")
    print(shap_df.to_string(index=False))

    # Predict quantiles (ensure quantile_models is available)
    if quantile_models is not None:
        scale_factor = 1e5  # Must match the factor used during quantile training
        quantile_preds = predict_quantiles(features_used, quantile_models, scale_factor=scale_factor)
        print("\nProbabilistic (Quantile) Predictions for the new data point:")
        for q in sorted(quantile_preds.keys()):
            print(f"Quantile {q}: {quantile_preds[q][0]:.2f}")

        print("\nAdvantages: Probabilistic forecasting provides uncertainty intervals instead of a single point estimate. This range helps decision-makers assess risk by indicating that the actual resale price is likely to lie within the specified quantile interval (e.g., between the 5th and 95th percentiles).")

        # Visualization: Plot quantile fan chart.
        plot_quantile_fan(quantile_preds, point_pred)
    else:
        print("Quantile models not found. Please ensure 'xgb_quantile_models.joblib' is available.")

    # (Optional) Deployment example for the point model:
    loaded_point_model = joblib.load("xgboost_house_price_model_propre.joblib")
    print("\nPoint model loaded for new prediction.")
    new_data = features_used.iloc[0:1]
    new_pred = loaded_point_model.predict(new_data)
    print("New data point prediction (Point Model): {:.2f}".format(new_pred[0]))
