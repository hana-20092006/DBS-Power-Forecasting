import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIRS = {
    "xgboost": Path("models/xgboost"),
    "lstm": Path("models/lstm"),
    "transformer": Path("models/transformer"),
}

OUTPUT_DIR = Path("models/ensemble")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

CITIES = [
    "chennai",
    "delhi",
]

MODELS = [
    "xgboost",
    "lstm",
    "transformer",
]


# ============================================================
# METRICS
# ============================================================

def calculate_smape(y_true, y_pred):

    denominator = (
        np.abs(y_true) +
        np.abs(y_pred)
    )

    numerator = (
        2 * np.abs(
            y_pred - y_true
        )
    )

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return (
        np.mean(
            numerator[mask] /
            denominator[mask]
        )
        * 100
    )


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    smape = calculate_smape(
        y_true,
        y_pred
    )

    return {
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "R2": r2,
        "sMAPE_percent": smape,
    }


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions(city, dataset):

    predictions = {}
    actual = None

    # --------------------------------------------------------
    # Load all model prediction files
    # --------------------------------------------------------

    for model in MODELS:

        file_path = (
            MODEL_DIRS[model] /
            f"{city}_{dataset}_predictions.csv"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing prediction file:\n"
                f"{file_path}"
            )

        df = pd.read_csv(file_path)

        required_columns = [
            "Actual_Power_MW",
            "Predicted_Power_MW",
        ]

        for column in required_columns:

            if column not in df.columns:

                raise ValueError(
                    f"Missing column '{column}' "
                    f"in {file_path}"
                )

        predictions[model] = df[
            "Predicted_Power_MW"
        ].to_numpy()

        # Store actual values from each file
        if actual is None:

            actual = df[
                "Actual_Power_MW"
            ].to_numpy()
    return actual, predictions
        # ----------------------------------------------------
        # Check that all models have compatible lengths
        # ----------------------------------------------------

# ========================================================
# ALIGN SEQUENCE MODELS WITH XGBOOST
# ========================================================
def align_predictions(
    city,
    dataset,
    actual,
    predictions
):
    """
    Align XGBoost, LSTM and Transformer predictions.

    All three prediction files are currently generated
    on the same timeline, so no 24-row trimming is needed.
    """

    print()
    print("Prediction alignment:")

    xgb_length = len(
        predictions["xgboost"]
    )

    lstm_length = len(
        predictions["lstm"]
    )

    transformer_length = len(
        predictions["transformer"]
    )

    actual_length = len(
        actual
    )

    print(
        f"  XGBoost      : {xgb_length}"
    )

    print(
        f"  LSTM         : {lstm_length}"
    )

    print(
        f"  Transformer  : {transformer_length}"
    )

    print(
        f"  Actual       : {actual_length}"
    )

    # --------------------------------------------------------
    # All prediction timelines must match
    # --------------------------------------------------------

    if not (
        xgb_length
        == lstm_length
        == transformer_length
        == actual_length
    ):

        raise ValueError(
            f"\nPrediction lengths do not match "
            f"for {city} {dataset}.\n"
            f"XGBoost      : {xgb_length}\n"
            f"LSTM         : {lstm_length}\n"
            f"Transformer  : {transformer_length}\n"
            f"Actual       : {actual_length}"
        )

    # --------------------------------------------------------
    # Final alignment
    # --------------------------------------------------------

    print()
    print("Final aligned lengths:")

    print(
        f"  XGBoost      : "
        f"{len(predictions['xgboost'])}"
    )

    print(
        f"  LSTM         : "
        f"{len(predictions['lstm'])}"
    )

    print(
        f"  Transformer  : "
        f"{len(predictions['transformer'])}"
    )

    print(
        f"  Actual       : "
        f"{len(actual)}"
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    final_lengths = {
        model: len(predictions[model])
        for model in MODELS
    }

    if len(
        set(final_lengths.values())
    ) != 1:

        raise ValueError(
            "Prediction lengths are still "
            "inconsistent."
        )

    if len(actual) != xgb_length:

        raise ValueError(
            "Actual values are not aligned "
            "with prediction lengths."
        )

    return actual, predictions
# ============================================================
# FIND WEIGHTS
# ============================================================

def find_best_weights(
    y_true,
    predictions
):

    best_weights = None
    best_mae = float("inf")

    # Search weights in increments of 0.05.
    # The three weights must sum to 1.

    for w_xgb in np.arange(
        0.0,
        1.01,
        0.05
    ):

        for w_lstm in np.arange(
            0.0,
            1.01 - w_xgb,
            0.05
        ):

            w_transformer = (
                1.0 -
                w_xgb -
                w_lstm
            )

            if w_transformer < -1e-9:
                continue

            ensemble_prediction = (

                w_xgb *
                predictions["xgboost"]

                +

                w_lstm *
                predictions["lstm"]

                +

                w_transformer *
                predictions["transformer"]
            )

            mae = mean_absolute_error(
                y_true,
                ensemble_prediction
            )

            if mae < best_mae:

                best_mae = mae

                best_weights = {
                    "xgboost": w_xgb,
                    "lstm": w_lstm,
                    "transformer":
                        w_transformer,
                }

    return best_weights


# ============================================================
# ENSEMBLE
# ============================================================

all_results = []


for city in CITIES:

    print()
    print("=" * 70)
    print(
        f"ENSEMBLE: {city.upper()}"
    )
    print("=" * 70)

    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    y_internal, internal_predictions = (
    load_predictions(
        city,
        "internal"
    )
)

    y_internal, internal_predictions = (
        align_predictions(
            city,
            "internal",
            y_internal,
            internal_predictions
        )
    )

    print()
    print("Finding ensemble weights")
    print("Using 2024 internal validation ONLY...")

    weights = find_best_weights(
        y_internal,
        internal_predictions
    )

    print()
    print("SELECTED WEIGHTS")
    print("-" * 70)

    for model in MODELS:

        print(
            f"{model.capitalize():15s}: "
            f"{weights[model]:.2f}"
        )

    # ========================================================
    # INTERNAL ENSEMBLE PREDICTION
    # ========================================================

    internal_ensemble = (

        weights["xgboost"] *
        internal_predictions["xgboost"]

        +

        weights["lstm"] *
        internal_predictions["lstm"]

        +

        weights["transformer"] *
        internal_predictions["transformer"]
    )

    internal_metrics = calculate_metrics(
        y_internal,
        internal_ensemble
    )

    print()
    print("2024 INTERNAL ENSEMBLE")
    print("-" * 70)

    print(
        f"MAE  : "
        f"{internal_metrics['MAE_MW']:.4f} MW"
    )

    print(
        f"RMSE : "
        f"{internal_metrics['RMSE_MW']:.4f} MW"
    )

    print(
        f"R²   : "
        f"{internal_metrics['R2']:.4f}"
    )

    print(
        f"sMAPE: "
        f"{internal_metrics['sMAPE_percent']:.2f}%"
    )

    # ========================================================
    # SAVE INTERNAL ENSEMBLE
    # ========================================================

    internal_df = pd.DataFrame({

        "Actual_Power_MW":
            y_internal,

        "Predicted_Power_MW":
            internal_ensemble,

    })

    internal_df.to_csv(
        OUTPUT_DIR /
        f"{city}_internal_ensemble_predictions.csv",
        index=False
    )

    # ========================================================
    # 2025 OOT
    # ========================================================

    y_oot, oot_predictions = (
    load_predictions(
        city,
        "oot"
    )
)

    y_oot, oot_predictions = (
        align_predictions(
            city,
            "oot",
            y_oot,
            oot_predictions
        )
    )

    # ========================================================
    # APPLY FROZEN WEIGHTS
    # ========================================================

    oot_ensemble = (

        weights["xgboost"] *
        oot_predictions["xgboost"]

        +

        weights["lstm"] *
        oot_predictions["lstm"]

        +

        weights["transformer"] *
        oot_predictions["transformer"]
    )

    oot_metrics = calculate_metrics(
        y_oot,
        oot_ensemble
    )

    print()
    print("2025 OUT-OF-TIME ENSEMBLE")
    print("-" * 70)

    print(
        f"MAE  : "
        f"{oot_metrics['MAE_MW']:.4f} MW"
    )

    print(
        f"RMSE : "
        f"{oot_metrics['RMSE_MW']:.4f} MW"
    )

    print(
        f"R²   : "
        f"{oot_metrics['R2']:.4f}"
    )

    print(
        f"sMAPE: "
        f"{oot_metrics['sMAPE_percent']:.2f}%"
    )

    # ========================================================
    # SAVE OOT ENSEMBLE PREDICTIONS
    # ========================================================

    oot_df = pd.DataFrame({

        "Actual_Power_MW":
            y_oot,

        "Predicted_Power_MW":
            oot_ensemble,

    })

    oot_df.to_csv(
        OUTPUT_DIR /
        f"{city}_oot_ensemble_predictions.csv",
        index=False
    )

    # ========================================================
    # SAVE WEIGHTS
    # ========================================================

    weights_df = pd.DataFrame([{

        "City": city,

        "XGBoost_weight":
            weights["xgboost"],

        "LSTM_weight":
            weights["lstm"],

        "Transformer_weight":
            weights["transformer"],

    }])

    weights_df.to_csv(
        OUTPUT_DIR /
        f"{city}_ensemble_weights.csv",
        index=False
    )

    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.append({

        "City": city,

        "Dataset": "2025 OOT",

        **oot_metrics,

        "XGBoost_weight":
            weights["xgboost"],

        "LSTM_weight":
            weights["lstm"],

        "Transformer_weight":
            weights["transformer"],

    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_DIR /
    "ensemble_results.csv",
    index=False
)


print()
print("=" * 70)
print("ENSEMBLE COMPLETE")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    f"Saved ensemble outputs to: "
    f"{OUTPUT_DIR}"
)