import pandas as pd
import numpy as np
import shap

from pathlib import Path
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models/xgboost")
DATA_DIR = Path("data/model_ready")
OUTPUT_DIR = Path("models/shap")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

CITIES = [
    "chennai",
    "delhi"
]

FEATURES = [

    # Weather
    "GHI",
    "DNI",
    "Temperature",
    "Cloud_Cover",
    "Wind_Speed",
    "Wind_Direction",
    "Wind_Gusts",
    "Pressure",
    "Humidity",
    "Precipitation",

    # Time
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",

    # Lag features
    "GHI_lag_1",
    "GHI_lag_24",

    "DNI_lag_1",
    "DNI_lag_24",

    "Temperature_lag_1",
    "Temperature_lag_24",

    "Wind_Speed_lag_1",
    "Wind_Speed_lag_24",

    "Humidity_lag_1",
    "Humidity_lag_24",

    "Cloud_Cover_lag_1",
    "Cloud_Cover_lag_24",

    "Actual_Power_MW_lag_1",
    "Actual_Power_MW_lag_24",

    # Rolling features
    "GHI_rolling_3h",
    "DNI_rolling_3h",
    "Temperature_rolling_3h",
    "Wind_Speed_rolling_3h",
    "Humidity_rolling_3h",
    "Cloud_Cover_rolling_3h",
    "Actual_Power_MW_rolling_3h",
]


# ============================================================
# SHAP ANALYSIS
# ============================================================

for city in CITIES:

    print()
    print("=" * 70)
    print(f"SHAP ANALYSIS: {city.upper()}")
    print("=" * 70)


    # --------------------------------------------------------
    # Load trained XGBoost model
    # --------------------------------------------------------

    model = XGBRegressor()

    model.load_model(
        MODEL_DIR /
        f"{city}_xgboost.json"
    )

    print("XGBoost model loaded.")


    # --------------------------------------------------------
    # Load 2025 OOT features
    # --------------------------------------------------------

    X_oot = np.load(
        DATA_DIR /
        city /
        "X_oot_xgb.npy"
    )

    print(
        f"2025 OOT data: {X_oot.shape}"
    )


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    X_oot_df = pd.DataFrame(
        X_oot,
        columns=FEATURES
    )


    # --------------------------------------------------------
    # Use a representative sample
    #
    # SHAP on the entire dataset is unnecessary and can
    # take longer.
    # --------------------------------------------------------

    sample_size = min(
        2000,
        len(X_oot_df)
    )

    X_sample = X_oot_df.sample(
        n=sample_size,
        random_state=42
    )

    print(
        f"SHAP sample size: {len(X_sample)}"
    )


    # --------------------------------------------------------
    # Create SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )


    print("Calculating SHAP values...")

    shap_values = explainer(
        X_sample
    )


    # ========================================================
    # SAVE SHAP VALUES
    # ========================================================

    shap_df = pd.DataFrame(
        shap_values.values,
        columns=FEATURES
    )

    shap_df.to_csv(
        OUTPUT_DIR /
        f"{city}_shap_values.csv",
        index=False
    )


    # ========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ========================================================

    mean_abs_shap = (
        np.abs(
            shap_values.values
        )
        .mean(axis=0)
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance_df = (
        importance_df
        .sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    importance_df.to_csv(
        OUTPUT_DIR /
        f"{city}_shap_feature_importance.csv",
        index=False
    )


    # ========================================================
    # SUMMARY PLOT
    # ========================================================

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    import matplotlib.pyplot as plt

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{city}_shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # BAR PLOT
    # ========================================================

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{city}_shap_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("Top SHAP features:")

    print(
        importance_df.head(10).to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved SHAP outputs to: {OUTPUT_DIR}"
    )


print()
print("=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)