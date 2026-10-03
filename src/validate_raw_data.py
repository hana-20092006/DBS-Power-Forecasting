import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/raw")

FILES = [
    "chennai_2024_weather.csv",
    "chennai_2025_weather.csv",
    "delhi_2024_weather.csv",
    "delhi_2025_weather.csv"
]


# ============================================================
# Validation function
# ============================================================

def validate_dataset(filename):

    path = DATA_DIR / filename

    print()
    print("=" * 70)
    print(f"VALIDATING: {filename}")
    print("=" * 70)

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    print("\n1. Dataset shape")
    print("----------------")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")


    # --------------------------------------------------------
    # Timestamp range
    # --------------------------------------------------------

    print("\n2. Timestamp range")
    print("------------------")

    print(f"First timestamp : {df['Timestamp'].min()}")
    print(f"Last timestamp  : {df['Timestamp'].max()}")


    # --------------------------------------------------------
    # Duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = df["Timestamp"].duplicated().sum()

    print("\n3. Duplicate timestamps")
    print("-----------------------")
    print(f"Duplicates: {duplicate_count}")


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n4. Missing values")
    print("-----------------")

    missing = df.isna().sum()

    print(missing)


    # --------------------------------------------------------
    # Negative values
    # --------------------------------------------------------

    print("\n5. Negative values")
    print("------------------")

    numeric_columns = [
        "GHI",
        "DNI",
        "Temperature",
        "Cloud_Cover",
        "Wind_Speed",
        "Wind_Gusts",
        "Pressure",
        "Humidity",
        "Precipitation"
    ]

    for column in numeric_columns:

        negative_count = (
            df[column] < 0
        ).sum()

        print(
            f"{column:20s}: "
            f"{negative_count}"
        )


    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\n6. Basic statistics")
    print("-------------------")

    print(
        df[numeric_columns]
        .describe()
        .round(2)
    )


    # --------------------------------------------------------
    # Important physical ranges
    # --------------------------------------------------------

    print("\n7. Important ranges")
    print("-------------------")

    print(
        f"GHI range          : "
        f"{df['GHI'].min():.2f} → "
        f"{df['GHI'].max():.2f}"
    )

    print(
        f"DNI range          : "
        f"{df['DNI'].min():.2f} → "
        f"{df['DNI'].max():.2f}"
    )

    print(
        f"Temperature range  : "
        f"{df['Temperature'].min():.2f} → "
        f"{df['Temperature'].max():.2f}"
    )

    print(
        f"Wind speed range   : "
        f"{df['Wind_Speed'].min():.2f} → "
        f"{df['Wind_Speed'].max():.2f}"
    )

    print(
        f"Humidity range     : "
        f"{df['Humidity'].min():.2f} → "
        f"{df['Humidity'].max():.2f}"
    )

    print(
        f"Cloud cover range  : "
        f"{df['Cloud_Cover'].min():.2f} → "
        f"{df['Cloud_Cover'].max():.2f}"
    )


    # --------------------------------------------------------
    # Solar nighttime check
    # --------------------------------------------------------

    nighttime_rows = (
        df["GHI"] == 0
    ).sum()

    print("\n8. Solar radiation")
    print("------------------")

    print(
        f"Rows with GHI = 0: "
        f"{nighttime_rows}"
    )


    # --------------------------------------------------------
    # Wind direction range
    # --------------------------------------------------------

    print("\n9. Wind direction")
    print("-----------------")

    print(
        f"Wind direction range: "
        f"{df['Wind_Direction'].min():.2f} → "
        f"{df['Wind_Direction'].max():.2f}"
    )


    print("\nValidation finished.")


# ============================================================
# Run validation for all datasets
# ============================================================

for file in FILES:

    validate_dataset(file)


print()
print("=" * 70)
print("ALL DATASETS VALIDATED")
print("=" * 70)