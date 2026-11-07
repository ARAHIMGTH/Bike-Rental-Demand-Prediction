
# ============================================================
# RIDEPREDICT - OPTIMIZED MODEL TRAINING
# Bike Rental Demand Prediction
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading Dataset...")

df = pd.read_csv("dataset/train.csv")

print("Dataset Shape:", df.shape)

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\nPreprocessing Data...")

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d/%m/%Y",
    errors="coerce"
)

df = df.dropna(subset=["Date"])

df = df.sort_values(
    by=["Date", "Hour"]
).reset_index(drop=True)

numeric_columns = df.select_dtypes(
    include=["number"]
).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

df["Season"] = df["Season"].fillna(
    df["Season"].mode()[0]
)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

print("\nCreating Features...")

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)

df["IsRushHour"] = df["Hour"].isin(
    [7, 8, 9, 17, 18, 19]
).astype(int)

df = pd.get_dummies(
    df,
    columns=["Season"],
    dtype=int
)

df = df.drop(
    columns=["Id", "Date"],
    errors="ignore"
)

# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Bikes_Rented"]
)

y = df["Bikes_Rented"]

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(
    X.median()
)

print("Total Features:", X.shape[1])

# ============================================================
# 5. TIME-BASED DATA SPLIT
# ============================================================

print("\nSplitting Dataset...")

# First 70%: Training
# Next 15%: Calibration
# Last 15%: Testing

train_end = int(len(X) * 0.70)
calibration_end = int(len(X) * 0.85)

X_train = X.iloc[:train_end]
X_calibration = X.iloc[train_end:calibration_end]
X_test = X.iloc[calibration_end:]

y_train = y.iloc[:train_end]
y_calibration = y.iloc[train_end:calibration_end]
y_test = y.iloc[calibration_end:]

print("Training Samples:", len(X_train))
print("Calibration Samples:", len(X_calibration))
print("Testing Samples:", len(X_test))

# ============================================================
# 6. LIGHTWEIGHT EXTRA TREES MODEL
# ============================================================

print("\nTraining Optimized Extra Trees Model...")

model = ExtraTreesRegressor(
    n_estimators=50,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model Training Completed!")

# ============================================================
# 7. CALIBRATE PREDICTION INTERVAL
# ============================================================

print("\nCalibrating Prediction Range...")

calibration_predictions = model.predict(
    X_calibration
)

absolute_errors = np.abs(
    y_calibration.values - calibration_predictions
)

# Approximate 90% absolute-error interval
interval_radius = np.quantile(
    absolute_errors,
    0.90
)

joblib.dump(
    float(interval_radius),
    "models/prediction_interval_radius.pkl"
)

print("90% Calibration Error Radius:", round(interval_radius, 2))

# ============================================================
# 8. MODEL EVALUATION
# ============================================================

print("\nEvaluating Model...")

predictions = model.predict(
    X_test
)

predictions = np.maximum(
    predictions,
    0
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("R2 Percentage:", round(r2 * 100, 2), "%")

# ============================================================
# 9. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/best_model.pkl",
    compress=3
)

joblib.dump(
    list(X.columns),
    "models/feature_names.pkl"
)

model_size = os.path.getsize(
    "models/best_model.pkl"
) / (1024 * 1024)

print("\nModel Saved Successfully!")

print("Optimized Model Size:", round(model_size, 2), "MB")

# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\nCreating Actual vs Predicted Graph...")

sample_size = min(
    100,
    len(y_test)
)

plt.figure(figsize=(12, 5))

plt.plot(
    range(sample_size),
    y_test.iloc[:sample_size].values,
    label="Actual",
    marker="o"
)

plt.plot(
    range(sample_size),
    predictions[:sample_size],
    label="Predicted",
    marker="x"
)

plt.title("Actual vs Predicted Bike Rentals")
plt.xlabel("Test Samples")
plt.ylabel("Bike Rentals")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "static/actual_vs_predicted.png",
    dpi=150
)

plt.close()

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\nGenerating Feature Importance...")

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

top_features = importance.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Factors Affecting Bike Rental Demand")
plt.xlabel("Feature Importance")
plt.tight_layout()

plt.savefig(
    "static/feature_importance.png",
    dpi=150
)

plt.close()

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 50)

print("RIDEPREDICT TRAINING COMPLETED")

print("=" * 50)

print("Algorithm: Extra Trees Regressor")
print("Trees: 50")
print("Maximum Depth: 12")
print("Model Size:", round(model_size, 2), "MB")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2 * 100, 2), "%")

print("\nSaved Files:")

print("models/best_model.pkl")
print("models/feature_names.pkl")
print("models/prediction_interval_radius.pkl")
print("static/actual_vs_predicted.png")
print("static/feature_importance.png")

print("\nAll Tasks Completed Successfully!")
