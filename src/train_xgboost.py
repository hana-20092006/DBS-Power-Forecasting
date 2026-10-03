import numpy as np
import pandas as pd

from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/model_ready")
OUTPUT_DIR = Path("models/xgboost")

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


# ============================================================
# METRICS
# ============================================================

def calculate_smape(
    y_true,
    y_pred
):

    denominator = (
        np.abs(y_true)
        +
        np.abs(y_pred)
    )

    numerator = (
        2 * np.abs(
            y_pred - y_true
        )
    )

    mask = denominator != 0

    return (
        np.mean(
            numerator[mask]
            /
            denominator[mask]
        )
        * 100
    )


def calculate_metrics(
    y_true,
    y_pred
):

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
# MODEL
# ============================================================

def create_model():

    return XGBRegressor(

        objective="reg:squarederror",

        n_estimators=500,

        learning_rate=0.05,

        max_depth=6,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.0,

        reg_lambda=1.0,

        random_state=42,

        n_jobs=-1,

    )


# ============================================================
# TRAIN EACH CITY
# ============================================================

all_results = []


for city in CITIES:

    print()
    print("=" * 70)
    print(
        f"XGBOOST TRAINING: {city.upper()}"
    )
    print("=" * 70)


    city_dir = (
        INPUT_DIR / city
    )


    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train = np.load(
        city_dir / "X_train_xgb.npy"
    )

    y_train = np.load(
        city_dir / "y_train_xgb.npy"
    )


    # --------------------------------------------------------
    # Load internal validation
    # --------------------------------------------------------

    X_internal = np.load(
        city_dir / "X_internal_val_xgb.npy"
    )

    y_internal = np.load(
        city_dir / "y_internal_val_xgb.npy"
    )


    # --------------------------------------------------------
    # Load 2025 OOT validation
    # --------------------------------------------------------

    X_oot = np.load(
        city_dir / "X_oot_xgb.npy"
    )

    y_oot = np.load(
        city_dir / "y_oot_xgb.npy"
    )


    print()
    print("Data:")

    print(
        f"  Train              : "
        f"{X_train.shape}"
    )

    print(
        f"  Internal validation: "
        f"{X_internal.shape}"
    )

    print(
        f"  2025 OOT validation: "
        f"{X_oot.shape}"
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()


    print()
    print("Training XGBoost...")


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_internal, y_internal)
        ],
        verbose=False,
    )


    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    internal_predictions = (
        model.predict(
            X_internal
        )
    )

    internal_metrics = (
        calculate_metrics(
            y_internal,
            internal_predictions
        )
    )


    # ========================================================
    # 2025 OUT-OF-TIME VALIDATION
    # ========================================================

    oot_predictions = (
        model.predict(
            X_oot
        )
    )

    oot_metrics = (
        calculate_metrics(
            y_oot,
            oot_predictions
        )
    )


    # --------------------------------------------------------
    # Print internal results
    # --------------------------------------------------------

    print()
    print(
        "2024 INTERNAL VALIDATION"
    )

    print(
        f"  MAE  : "
        f"{internal_metrics['MAE_MW']:.4f} MW"
    )

    print(
        f"  RMSE : "
        f"{internal_metrics['RMSE_MW']:.4f} MW"
    )

    print(
        f"  R²   : "
        f"{internal_metrics['R2']:.4f}"
    )

    print(
        f"  sMAPE: "
        f"{internal_metrics['sMAPE_percent']:.2f}%"
    )


    # --------------------------------------------------------
    # Print OOT results
    # --------------------------------------------------------

    print()
    print(
        "2025 OUT-OF-TIME VALIDATION"
    )

    print(
        f"  MAE  : "
        f"{oot_metrics['MAE_MW']:.4f} MW"
    )

    print(
        f"  RMSE : "
        f"{oot_metrics['RMSE_MW']:.4f} MW"
    )

    print(
        f"  R²   : "
        f"{oot_metrics['R2']:.4f}"
    )

    print(
        f"  sMAPE: "
        f"{oot_metrics['sMAPE_percent']:.2f}%"
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        OUTPUT_DIR /
        f"{city}_xgboost.json"
    )

    model.save_model(
        model_path
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------
    # ============================================================
# SAVE INTERNAL VALIDATION PREDICTIONS
# ============================================================

    internal_prediction_df = pd.DataFrame({
    "Actual_Power_MW": y_internal,
    "Predicted_Power_MW": internal_predictions
    })

    internal_prediction_path = (
        OUTPUT_DIR / f"{city}_internal_predictions.csv"
    )

    internal_prediction_df.to_csv(
        internal_prediction_path,
        index=False
    )

    print(
        f"Internal predictions saved: "
        f"{internal_prediction_path}"
    )

    internal_prediction_df.to_csv(
        internal_prediction_path,
        index=False
    )

    print(
        f"Internal predictions saved: "
        f"{internal_prediction_path}"
    )
    predictions_df = pd.DataFrame({

        "Actual_Power_MW":
            y_oot,

        "Predicted_Power_MW":
            oot_predictions,

    })


    predictions_path = (
        OUTPUT_DIR /
        f"{city}_oot_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    all_results.append({

        "City": city,

        "Dataset":
            "2025 OOT",

        **oot_metrics,

    })


    print()
    print(
        f"Model saved: "
        f"{model_path}"
    )

    print(
        f"Predictions saved: "
        f"{predictions_path}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    OUTPUT_DIR /
    "xgboost_results.csv",
    index=False
)


print()
print("=" * 70)
print("XGBOOST TRAINING COMPLETE")
print("=" * 70)

print()
print(results_df.to_string(
    index=False
))