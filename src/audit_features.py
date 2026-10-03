import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/splits")


FILES = [
    "chennai_train_2024.csv",
    "chennai_internal_validation_2024.csv",
    "chennai_out_of_time_validation_2025.csv",

    "delhi_train_2024.csv",
    "delhi_internal_validation_2024.csv",
    "delhi_out_of_time_validation_2025.csv",
]


# ============================================================
# EXPECTED FEATURES
# ============================================================

BASE_FEATURES = [
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
]

TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
]

LAG_FEATURES = [
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
]

ROLLING_FEATURES = [
    "GHI_rolling_3h",
    "DNI_rolling_3h",
    "Temperature_rolling_3h",
    "Wind_Speed_rolling_3h",
    "Humidity_rolling_3h",
    "Cloud_Cover_rolling_3h",
    "Actual_Power_MW_rolling_3h",
]

TARGET = "Actual_Power_MW"


EXPECTED_FEATURES = (
    BASE_FEATURES
    + TIME_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)


# ============================================================
# AUDIT
# ============================================================

print("=" * 70)
print("FEATURE AUDIT")
print("=" * 70)

print()
print(
    f"Expected input features : "
    f"{len(EXPECTED_FEATURES)}"
)

print(
    f"Expected target         : "
    f"{TARGET}"
)


for filename in FILES:

    print()
    print("=" * 70)
    print(filename)
    print("=" * 70)

    path = DATA_DIR / filename

    df = pd.read_csv(path)

    actual_features = [
        column
        for column in EXPECTED_FEATURES
        if column in df.columns
    ]

    missing_features = [
        column
        for column in EXPECTED_FEATURES
        if column not in df.columns
    ]

    unexpected_features = [
        column
        for column in df.columns
        if (
            column not in EXPECTED_FEATURES
            and column not in [
                "Timestamp",
                TARGET,
                "Solar_Power_MW",
                "Wind_Power_MW",
            ]
        )
    ]


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print(
        f"Rows                  : {len(df)}"
    )

    print(
        f"Columns               : {len(df.columns)}"
    )

    print(
        f"Expected features     : "
        f"{len(actual_features)}/"
        f"{len(EXPECTED_FEATURES)}"
    )

    print(
        f"Target present        : "
        f"{TARGET in df.columns}"
    )

    print(
        f"Missing feature count : "
        f"{len(missing_features)}"
    )

    print(
        f"Unexpected features   : "
        f"{len(unexpected_features)}"
    )


    if missing_features:

        print()
        print("MISSING FEATURES:")

        for feature in missing_features:
            print(
                f"  - {feature}"
            )


    if unexpected_features:

        print()
        print("UNEXPECTED FEATURES:")

        for feature in unexpected_features:
            print(
                f"  - {feature}"
            )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        df[
            EXPECTED_FEATURES + [TARGET]
        ]
        .isna()
        .sum()
        .sum()
    )

    print()
    print(
        f"Missing values in ML data: "
        f"{missing_values}"
    )


    # --------------------------------------------------------
    # Target statistics
    # --------------------------------------------------------

    print()
    print("Target statistics:")

    print(
        f"  Mean : "
        f"{df[TARGET].mean():.4f} MW"
    )

    print(
        f"  Min  : "
        f"{df[TARGET].min():.4f} MW"
    )

    print(
        f"  Max  : "
        f"{df[TARGET].max():.4f} MW"
    )


print()
print("=" * 70)
print("FEATURE AUDIT COMPLETE")
print("=" * 70)