import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/model_ready")
SCALER_DIR = OUTPUT_DIR / "scalers"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SCALER_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

CITIES = ["chennai", "delhi"]

TARGET = "Actual_Power_MW"

SEQUENCE_LENGTH = 24


# ============================================================
# FEATURES
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
# HELPER FUNCTIONS
# ============================================================

def load_dataset(filename):

    path = INPUT_DIR / filename

    df = pd.read_csv(path)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    df = df.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    return df


def create_sequences(X, y, sequence_length):

    X_sequences = []
    y_sequences = []

    for i in range(
        sequence_length,
        len(X)
    ):

        X_sequences.append(
            X[
                i - sequence_length:i
            ]
        )

        y_sequences.append(
            y[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


def save_numpy(path, array):

    np.save(
        path,
        array
    )


# ============================================================
# PROCESS EACH CITY
# ============================================================

for city in CITIES:

    print()
    print("=" * 70)
    print(
        f"PREPARING MODEL DATA: {city.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_df = load_dataset(
        f"{city}_train_2024.csv"
    )

    internal_val_df = load_dataset(
        f"{city}_internal_validation_2024.csv"
    )

    oot_df = load_dataset(
        f"{city}_out_of_time_validation_2025.csv"
    )


    print()
    print("Raw split sizes:")

    print(
        f"  Train              : "
        f"{len(train_df)}"
    )

    print(
        f"  Internal validation: "
        f"{len(internal_val_df)}"
    )

    print(
        f"  2025 OOT validation: "
        f"{len(oot_df)}"
    )


    # --------------------------------------------------------
    # Extract X and y
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ].values

    y_train = train_df[
        TARGET
    ].values

    X_internal_val = internal_val_df[
        FEATURES
    ].values

    y_internal_val = internal_val_df[
        TARGET
    ].values

    X_oot = oot_df[
        FEATURES
    ].values

    y_oot = oot_df[
        TARGET
    ].values


    # --------------------------------------------------------
    # Fit feature scaler ONLY on training data
    # --------------------------------------------------------

    feature_scaler = StandardScaler()

    X_train_scaled = (
        feature_scaler.fit_transform(
            X_train
        )
    )

    X_internal_val_scaled = (
        feature_scaler.transform(
            X_internal_val
        )
    )

    X_oot_scaled = (
        feature_scaler.transform(
            X_oot
        )
    )


    # --------------------------------------------------------
    # Target scaler
    #
    # Fit ONLY on training target.
    # --------------------------------------------------------

    target_scaler = StandardScaler()

    y_train_scaled = (
        target_scaler.fit_transform(
            y_train.reshape(-1, 1)
        )
        .ravel()
    )

    y_internal_val_scaled = (
        target_scaler.transform(
            y_internal_val.reshape(-1, 1)
        )
        .ravel()
    )

    y_oot_scaled = (
        target_scaler.transform(
            y_oot.reshape(-1, 1)
        )
        .ravel()
    )


    # ========================================================
    # XGBOOST DATA
    #
    # 2D:
    # samples × features
    # ========================================================

    city_dir = (
        OUTPUT_DIR / city
    )

    city_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    save_numpy(
        city_dir / "X_train_xgb.npy",
        X_train_scaled
    )

    save_numpy(
        city_dir / "y_train_xgb.npy",
        y_train
    )

    save_numpy(
        city_dir / "X_internal_val_xgb.npy",
        X_internal_val_scaled
    )

    save_numpy(
        city_dir / "y_internal_val_xgb.npy",
        y_internal_val
    )

    save_numpy(
        city_dir / "X_oot_xgb.npy",
        X_oot_scaled
    )

    save_numpy(
        city_dir / "y_oot_xgb.npy",
        y_oot
    )


    # ========================================================
    # NEURAL NETWORK DATA
    #
    # 3D:
    #
    # samples × sequence length × features
    # ========================================================

    X_train_seq, y_train_seq = (
        create_sequences(
            X_train_scaled,
            y_train_scaled,
            SEQUENCE_LENGTH
        )
    )


    # --------------------------------------------------------
    # Internal validation sequences
    #
    # Include the final 24 rows of the training period
    # as historical context so that the first validation
    # timestamp can also be predicted.
    #
    # IMPORTANT:
    # The target values still come ONLY from the
    # internal validation period.
    # --------------------------------------------------------

    X_internal_val_with_context = np.concatenate(
        [
            X_train_scaled[-SEQUENCE_LENGTH:],
            X_internal_val_scaled
        ],
        axis=0
    )

    X_internal_val_seq = []

    y_internal_val_seq = []

    for i in range(
        SEQUENCE_LENGTH,
        len(X_internal_val_with_context)
    ):

        X_internal_val_seq.append(
            X_internal_val_with_context[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y_internal_val_seq.append(
            y_internal_val_scaled[i - SEQUENCE_LENGTH]
        )

    X_internal_val_seq = np.array(
        X_internal_val_seq
    )

    y_internal_val_seq = np.array(
        y_internal_val_seq
    )
    

    # --------------------------------------------------------
    # 2025 OOT sequences
    #
    # Use the final 24 rows of 2024 internal validation
    # as historical context for the first 2025 prediction.
    # --------------------------------------------------------

    X_oot_with_context = np.concatenate(
        [
            X_internal_val_scaled[-SEQUENCE_LENGTH:],
            X_oot_scaled
        ],
        axis=0
    )

    X_oot_seq = []

    y_oot_seq = []

    for i in range(
        SEQUENCE_LENGTH,
        len(X_oot_with_context)
    ):

        X_oot_seq.append(
            X_oot_with_context[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y_oot_seq.append(
            y_oot_scaled[i - SEQUENCE_LENGTH]
        )

    X_oot_seq = np.array(
        X_oot_seq
    )

    y_oot_seq = np.array(
        y_oot_seq
    )


    save_numpy(
        city_dir / "X_train_nn.npy",
        X_train_seq
    )

    save_numpy(
        city_dir / "y_train_nn.npy",
        y_train_seq
    )

    save_numpy(
        city_dir / "X_internal_val_nn.npy",
        X_internal_val_seq
    )

    save_numpy(
        city_dir / "y_internal_val_nn.npy",
        y_internal_val_seq
    )

    save_numpy(
        city_dir / "X_oot_nn.npy",
        X_oot_seq
    )

    save_numpy(
        city_dir / "y_oot_nn.npy",
        y_oot_seq
    )


    # ========================================================
    # SAVE SCALERS
    # ========================================================

    with open(
        SCALER_DIR / f"{city}_feature_scaler.pkl",
        "wb"
    ) as f:

        pickle.dump(
            feature_scaler,
            f
        )


    with open(
        SCALER_DIR / f"{city}_target_scaler.pkl",
        "wb"
    ) as f:

        pickle.dump(
            target_scaler,
            f
        )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("Feature matrix:")
    print(
        f"  Number of features: "
        f"{X_train.shape[1]}"
    )

    print()
    print("XGBoost shapes:")

    print(
        f"  Train              : "
        f"{X_train_scaled.shape}"
    )

    print(
        f"  Internal validation: "
        f"{X_internal_val_scaled.shape}"
    )

    print(
        f"  2025 OOT validation: "
        f"{X_oot_scaled.shape}"
    )

    print()
    print("Neural network shapes:")

    print(
        f"  Train              : "
        f"{X_train_seq.shape}"
    )

    print(
        f"  Internal validation: "
        f"{X_internal_val_seq.shape}"
    )

    print(
        f"  2025 OOT validation: "
        f"{X_oot_seq.shape}"
    )

    print()
    print(
        "Scaler fitted ONLY on 2024 training data."
    )

    print(
        f"Saved model-ready data to: "
        f"{city_dir}"
    )


print()
print("=" * 70)
print("MODEL DATA PREPARATION COMPLETE")
print("=" * 70)