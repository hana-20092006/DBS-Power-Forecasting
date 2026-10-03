import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/features")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FILES = [
    "chennai_2024_power.csv",
    "chennai_2025_power.csv",
    "delhi_2024_power.csv",
    "delhi_2025_power.csv",
]


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(df):

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    df = df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Basic time features
    # --------------------------------------------------------

    df["hour"] = (
        df["Timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["Timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["Timestamp"].dt.month
    )


    # --------------------------------------------------------
    # Cyclical hour encoding
    # --------------------------------------------------------

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )


    # --------------------------------------------------------
    # Cyclical month encoding
    # --------------------------------------------------------

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )


    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    lag_columns = [
        "GHI",
        "DNI",
        "Temperature",
        "Wind_Speed",
        "Humidity",
        "Cloud_Cover",
        "Actual_Power_MW",
    ]

    for column in lag_columns:

        df[f"{column}_lag_1"] = (
            df[column].shift(1)
        )

        df[f"{column}_lag_24"] = (
            df[column].shift(24)
        )


    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    rolling_columns = [
        "GHI",
        "DNI",
        "Temperature",
        "Wind_Speed",
        "Humidity",
        "Cloud_Cover",
        "Actual_Power_MW",
    ]

    for column in rolling_columns:

        df[f"{column}_rolling_3h"] = (
            df[column]
            .shift(1)
            .rolling(window=3)
            .mean()
        )


    # --------------------------------------------------------
    # Remove rows created by lag/rolling operations
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "GHI_lag_24",
            "Wind_Speed_lag_24",
            "Actual_Power_MW_lag_24",
            "Actual_Power_MW_rolling_3h",
        ]
    ).reset_index(
        drop=True
    )


    return df


# ============================================================
# PROCESS ALL DATASETS
# ============================================================

for filename in FILES:

    input_path = (
        INPUT_DIR / filename
    )

    print()
    print("=" * 70)
    print(f"FEATURE ENGINEERING: {filename}")
    print("=" * 70)

    df = pd.read_csv(
        input_path
    )

    print(
        f"Input rows    : {len(df)}"
    )

    print(
        f"Input columns : {len(df.columns)}"
    )


    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    df = create_features(
        df
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_filename = filename.replace(
        "_power.csv",
        "_features.csv"
    )

    output_path = (
        OUTPUT_DIR /
        output_filename
    )

    df.to_csv(
        output_path,
        index=False
    )


    print(
        f"Output rows   : {len(df)}"
    )

    print(
        f"Output columns: {len(df.columns)}"
    )

    print(
        f"Rows removed  : "
        f"{8784 - len(df) if '2024' in filename else 8760 - len(df)}"
    )

    print(
        f"Saved: {output_path}"
    )


print()
print("=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)