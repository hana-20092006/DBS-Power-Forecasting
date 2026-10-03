import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ENSEMBLE_DIR = Path("models/ensemble")
SHAP_DIR = Path("models/shap")
OUTPUT_DIR = Path("models/visualizations")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CITIES = [
    "chennai",
    "delhi"
]


# ============================================================
# HELPER
# ============================================================

def save_plot(filename):
    path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 1. ACTUAL VS PREDICTED
# ============================================================

for city in CITIES:

    print()
    print("=" * 70)
    print(f"ACTUAL VS PREDICTED: {city.upper()}")
    print("=" * 70)

    path = (
        ENSEMBLE_DIR /
        f"{city}_oot_ensemble_predictions.csv"
    )

    df = pd.read_csv(path)

    # Show first 500 observations for readability
    plot_df = df.iloc[:500]

    plt.figure(figsize=(14, 6))

    plt.plot(
        plot_df["Actual_Power_MW"],
        label="Actual Power"
    )

    plt.plot(
        plot_df["Predicted_Power_MW"],
        label="Predicted Power"
    )

    plt.xlabel("Time step")
    plt.ylabel("Power (MW)")

    plt.title(
        f"{city.title()} — Actual vs Predicted Solar Power"
    )

    plt.legend()

    save_plot(
        f"{city}_actual_vs_predicted.png"
    )


# ============================================================
# 2. MODEL PERFORMANCE COMPARISON
# ============================================================

performance_rows = []

for city in CITIES:

    # XGBoost
    xgb_results = pd.read_csv(
        "models/xgboost/xgboost_results.csv"
    )

    xgb_row = xgb_results[
        xgb_results["City"] == city
    ].iloc[0]

    # LSTM
    lstm_results = pd.read_csv(
        "models/lstm/lstm_results.csv"
    )

    lstm_row = lstm_results[
        lstm_results["City"] == city
    ].iloc[0]

    # Transformer
    transformer_results = pd.read_csv(
        "models/transformer/transformer_results.csv"
    )

    transformer_row = transformer_results[
        transformer_results["City"] == city
    ].iloc[0]

    performance_rows.extend([
        {
            "City": city.title(),
            "Model": "XGBoost",
            "MAE": xgb_row["MAE_MW"],
            "RMSE": xgb_row["RMSE_MW"],
            "R2": xgb_row["R2"],
        },
        {
            "City": city.title(),
            "Model": "LSTM",
            "MAE": lstm_row["MAE_MW"],
            "RMSE": lstm_row["RMSE_MW"],
            "R2": lstm_row["R2"],
        },
        {
            "City": city.title(),
            "Model": "Transformer",
            "MAE": transformer_row["MAE_MW"],
            "RMSE": transformer_row["RMSE_MW"],
            "R2": transformer_row["R2"],
        }
    ])


performance = pd.DataFrame(
    performance_rows
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for city in CITIES:

    city_name = city.title()

    subset = performance[
        performance["City"] == city_name
    ]

    plt.bar(
        [
            f"{city_name}\n{model}"
            for model in subset["Model"]
        ],
        subset["MAE"]
    )

plt.ylabel("MAE (MW)")
plt.title(
    "Model Comparison — Mean Absolute Error"
)

save_plot(
    "model_comparison_mae.png"
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

labels = []
values = []

for _, row in performance.iterrows():

    labels.append(
        f"{row['City']}\n{row['Model']}"
    )

    values.append(
        row["RMSE"]
    )

plt.bar(
    labels,
    values
)

plt.ylabel("RMSE (MW)")
plt.title(
    "Model Comparison — Root Mean Squared Error"
)

save_plot(
    "model_comparison_rmse.png"
)


# ------------------------------------------------------------
# R2
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

labels = []
values = []

for _, row in performance.iterrows():

    labels.append(
        f"{row['City']}\n{row['Model']}"
    )

    values.append(
        row["R2"]
    )

plt.bar(
    labels,
    values
)

plt.ylabel("R²")
plt.title(
    "Model Comparison — R² Score"
)

save_plot(
    "model_comparison_r2.png"
)


# ============================================================
# 3. ENSEMBLE PERFORMANCE
# ============================================================

ensemble_results = pd.read_csv(
    ENSEMBLE_DIR /
    "ensemble_results.csv"
)


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    ensemble_results["City"].str.title(),
    ensemble_results["MAE_MW"]
)

plt.ylabel("MAE (MW)")
plt.title(
    "Ensemble Performance — MAE"
)

save_plot(
    "ensemble_mae.png"
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    ensemble_results["City"].str.title(),
    ensemble_results["RMSE_MW"]
)

plt.ylabel("RMSE (MW)")
plt.title(
    "Ensemble Performance — RMSE"
)

save_plot(
    "ensemble_rmse.png"
)


# ------------------------------------------------------------
# R2
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    ensemble_results["City"].str.title(),
    ensemble_results["R2"]
)

plt.ylabel("R²")
plt.title(
    "Ensemble Performance — R²"
)

save_plot(
    "ensemble_r2.png"
)


# ============================================================
# 4. ERROR DISTRIBUTION
# ============================================================

for city in CITIES:

    path = (
        ENSEMBLE_DIR /
        f"{city}_oot_ensemble_predictions.csv"
    )

    df = pd.read_csv(path)

    error = (
        df["Predicted_Power_MW"]
        -
        df["Actual_Power_MW"]
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        error,
        bins=50
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Prediction Error (MW)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{city.title()} — Ensemble Prediction Error Distribution"
    )

    save_plot(
        f"{city}_error_distribution.png"
    )


# ============================================================
# 5. SHAP FEATURE IMPORTANCE
# ============================================================

for city in CITIES:

    print()
    print("=" * 70)
    print(f"SHAP VISUALIZATION: {city.upper()}")
    print("=" * 70)

    # Find SHAP CSV
    possible_files = list(
        SHAP_DIR.glob(
            f"{city}*.csv"
        )
    )

    if not possible_files:

        print(
            f"No SHAP CSV found for {city}"
        )

        continue

    print(
        f"Using SHAP file: "
        f"{possible_files[0]}"
    )

    shap_df = pd.read_csv(
        possible_files[0]
    )

    # Expected columns:
    # Feature
    # Mean_Absolute_SHAP

    if (
        "Feature" not in shap_df.columns
        or
        "Mean_Absolute_SHAP"
        not in shap_df.columns
    ):

        print(
            "SHAP CSV does not contain "
            "expected columns."
        )

        continue

    shap_df = (
        shap_df
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=True
        )
        .tail(10)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        shap_df["Feature"],
        shap_df["Mean_Absolute_SHAP"]
    )

    plt.xlabel(
        "Mean Absolute SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{city.title()} — Top 10 SHAP Features"
    )

    save_plot(
        f"{city}_shap_importance.png"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print()
print(
    f"All plots saved to: {OUTPUT_DIR}"
)