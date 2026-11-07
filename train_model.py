
# ============================================================
# BIKE RENTAL DEMAND PREDICTION
# Complete Machine Learning Training Pipeline
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\nLoading Dataset...")

df = pd.read_csv("dataset/train.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

print("\nStarting Data Preprocessing...")

# Convert Date into datetime format
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d/%m/%Y",
    errors="coerce"
)

# Remove rows with invalid dates
df = df.dropna(subset=["Date"])

# Sort data chronologically
df = df.sort_values(
    by=["Date", "Hour"]
).reset_index(drop=True)

# Handle missing values
numeric_columns = df.select_dtypes(
    include=["number"]
).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

# Handle missing categorical values
df["Season"] = df["Season"].fillna(
    df["Season"].mode()[0]
)

# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================

print("\nCreating New Features...")

# Calendar features
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Weekend feature
df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)

# Rush hour feature
df["IsRushHour"] = df["Hour"].isin(
    [7, 8, 9, 17, 18, 19]
).astype(int)

# Convert categorical season into numerical columns
df = pd.get_dummies(
    df,
    columns=["Season"],
    dtype=int
)

# Remove unnecessary columns
df = df.drop(
    columns=["Id", "Date"],
    errors="ignore"
)

# ============================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# ============================================================

print("\nSeparating Features and Target...")

X = df.drop(
    columns=["Bikes_Rented"]
)

y = df["Bikes_Rented"]

# Ensure all features are numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Fill any remaining missing values
X = X.fillna(
    X.median()
)

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Column:")
print(y.name)

print("\nTotal Target Values:")
print(len(y))

# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================

print("\nPerforming Train-Test Split...")

# Time-based split: first 80% for training
split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print("\nTraining Features:")
print(X_train.shape)

print("\nTraining Target:")
print(y_train.shape)

print("\nTesting Features:")
print(X_test.shape)

print("\nTesting Target:")
print(y_test.shape)

# ============================================================
# STEP 6: DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\nStarting Machine Learning Training...")

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        subsample=0.8,
        loss="huber",
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )
}

# ============================================================
# STEP 7: TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 45)
    print("Training:", name)
    print("=" * 45)

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predict testing data
    predictions = model.predict(
        X_test
    )

    # Bike rentals cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    # Calculate evaluation metrics
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

    # Store trained model
    trained_models[name] = model

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# ============================================================
# STEP 8: MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# STEP 9: SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_r2 = results_df.iloc[0]["R2 Score"]

best_rmse = results_df.iloc[0]["RMSE"]

print("\n" + "=" * 60)
print("BEST MODEL:", best_model_name)
print("BEST R2 SCORE:", round(best_r2, 4))
print("BEST RMSE:", round(best_rmse, 2))
print("=" * 60)

# ============================================================
# STEP 10: SAVE BEST MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    list(X_train.columns),
    "models/feature_names.pkl"
)

print("\nBest Model Saved Successfully!")

# ============================================================
# STEP 11: ACTUAL VS PREDICTED GRAPH
# ============================================================

print("\nCreating Actual vs Predicted Graph...")

best_predictions = best_model.predict(
    X_test
)

best_predictions = np.maximum(
    best_predictions,
    0
)

# Display first 100 test samples
sample_size = min(
    100,
    len(y_test)
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    range(sample_size),
    y_test.iloc[:sample_size].values,
    marker="o",
    label="Actual Bike Rentals"
)

plt.plot(
    range(sample_size),
    best_predictions[:sample_size],
    marker="x",
    label="Predicted Bike Rentals"
)

plt.title(
    "Bike Rental Demand: Actual vs Predicted"
)

plt.xlabel(
    "Test Data Samples"
)

plt.ylabel(
    "Number of Bikes Rented"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "models/actual_vs_predicted.png",
    dpi=300
)

plt.show()

print(
    "Actual vs Predicted Graph Saved Successfully!"
)

# ============================================================
# STEP 12: FEATURE IMPORTANCE ANALYSIS
# ============================================================

print("\nGenerating Feature Importance Analysis...")

# Use native feature importance when available
if hasattr(
    best_model,
    "feature_importances_"
):

    importance_values = (
        best_model.feature_importances_
    )

else:

    # Permutation importance for models without
    # native feature importance
    from sklearn.inspection import permutation_importance

    importance_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=3,
        random_state=42,
        scoring="r2",
        n_jobs=-1
    )

    importance_values = (
        importance_result.importances_mean
    )

feature_importance = pd.DataFrame({

    "Feature": X_train.columns,

    "Importance": importance_values

})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)

# Save feature importance
feature_importance.to_csv(
    "models/feature_importance.csv",
    index=False
)

# Plot top 10 features
top_features = feature_importance.head(
    10
).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(
    figsize=(12, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top 10 Factors Affecting Bike Rental Demand"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Features"
)

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png",
    dpi=300
)

plt.show()

print(
    "Feature Importance Graph Saved Successfully!"
)

# ============================================================
# STEP 13: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BIKE RENTAL DEMAND PREDICTION COMPLETED")
print("=" * 60)

print("Dataset Size:", len(df))

print("Total Features:", X.shape[1])

print("Training Samples:", len(X_train))

print("Testing Samples:", len(X_test))

print("Best Model:", best_model_name)

print("Final MAE:", round(
    mean_absolute_error(
        y_test,
        best_predictions
    ),
    2
))

print("Final RMSE:", round(
    np.sqrt(
        mean_squared_error(
            y_test,
            best_predictions
        )
    ),
    2
))

print("Final R2 Score:", round(
    r2_score(
        y_test,
        best_predictions
    ),
    4
))

print("\nModel Saved At:")
print("models/best_model.pkl")

print("\nFeature Names Saved At:")
print("models/feature_names.pkl")

print("\nAll Tasks Completed Successfully!")
