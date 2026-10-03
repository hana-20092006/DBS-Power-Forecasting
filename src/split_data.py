import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/splits")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASETS = {
    "chennai": {
        "train": "chennai_2024_features.csv",
        "validation": "chennai_2025_features.csv",
    },

    "delhi": {
        "train": "delhi_2024_features.csv",
        "validation": "delhi_2025_features.csv",
    }
}


# ============================================================
# PROCESS EACH CITY
# ============================================================

for city, files in DATASETS.items():

    print()
    print("=" * 70)
    print(f"CREATING CHRONOLOGICAL SPLIT: {city.upper()}")
    print("=" * 70)


    # --------------------------------------------------------
    # Load 2024
    # --------------------------------------------------------

    train_df = pd.read_csv(
        INPUT_DIR / files["train"]
    )

    # --------------------------------------------------------
    # Load 2025
    # --------------------------------------------------------

    validation_df = pd.read_csv(
        INPUT_DIR / files["validation"]
    )


    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    train_df["Timestamp"] = pd.to_datetime(
        train_df["Timestamp"]
    )

    validation_df["Timestamp"] = pd.to_datetime(
        validation_df["Timestamp"]
    )


    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    train_df = train_df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    validation_df = validation_df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Internal validation split
    #
    # Last 20% of 2024 is used for model selection.
    #
    # IMPORTANT:
    # This is NOT the 2025 out-of-time validation.
    # --------------------------------------------------------

    split_index = int(
        len(train_df) * 0.80
    )

    internal_train = (
        train_df.iloc[:split_index]
        .copy()
    )

    internal_validation = (
        train_df.iloc[split_index:]
        .copy()
    )


    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    internal_train.to_csv(
        OUTPUT_DIR /
        f"{city}_train_2024.csv",
        index=False
    )

    internal_validation.to_csv(
        OUTPUT_DIR /
        f"{city}_internal_validation_2024.csv",
        index=False
    )

    validation_df.to_csv(
        OUTPUT_DIR /
        f"{city}_out_of_time_validation_2025.csv",
        index=False
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()

    print("2024 total:")
    print(
        f"  Rows: {len(train_df)}"
    )

    print()

    print("Internal training:")
    print(
        f"  Rows: {len(internal_train)}"
    )

    print(
        f"  From: "
        f"{internal_train['Timestamp'].min()}"
    )

    print(
        f"  To  : "
        f"{internal_train['Timestamp'].max()}"
    )

    print()

    print("Internal validation:")
    print(
        f"  Rows: {len(internal_validation)}"
    )

    print(
        f"  From: "
        f"{internal_validation['Timestamp'].min()}"
    )

    print(
        f"  To  : "
        f"{internal_validation['Timestamp'].max()}"
    )

    print()

    print("2025 out-of-time validation:")
    print(
        f"  Rows: {len(validation_df)}"
    )

    print(
        f"  From: "
        f"{validation_df['Timestamp'].min()}"
    )

    print(
        f"  To  : "
        f"{validation_df['Timestamp'].max()}"
    )


print()
print("=" * 70)
print("CHRONOLOGICAL SPLITS COMPLETE")
print("=" * 70)