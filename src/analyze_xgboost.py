import numpy as np
import pandas as pd

from pathlib import Path

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models/xgboost")
DATA_DIR = Path("data/model_ready")


CITIES = [
    "chennai",
    "delhi",
]


# ============================================================
# FEATURE NAMES
# ============================================================

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
# ANALYZE
# ============================================================

for city in CITIES:

    print()
    print("=" * 70)
    print(
        f"XGBOOST FEATURE IMPORTANCE: {city.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = XGBRegressor()

    model.load_model(
        MODEL_DIR /
        f"{city}_xgboost.json"
    )


    # --------------------------------------------------------
    # Get importance
    # --------------------------------------------------------

    importance = (
        model.feature_importances_
    )
    if len(FEATURES) != len(importance):
        raise ValueError(
            f"Feature mismatch: "
            f"{len(FEATURES)} feature names but "
            f"{len(importance)} importance values."
        )

    importance_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": importance,

    })


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Print top 15
    # --------------------------------------------------------

    print()
    print(
        "TOP 15 FEATURES"
    )
    print(
        "-" * 70
    )

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Print lag feature importance
    # --------------------------------------------------------

    lag_features = (
        importance_df[
            importance_df["Feature"].str.contains(
                "_lag_|rolling"
            )
        ]
    )


    print()
    print(
        "TEMPORAL FEATURES"
    )
    print(
        "-" * 70
    )

    print(
        lag_features.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        MODEL_DIR /
        f"{city}_feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False
    )


    print()
    print(
        f"Saved: {output_path}"
    )


print()
print("=" * 70)
print("XGBOOST FEATURE ANALYSIS COMPLETE")
print("=" * 70)