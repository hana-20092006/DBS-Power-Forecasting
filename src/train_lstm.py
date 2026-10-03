import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/model_ready")
OUTPUT_DIR = Path("models/lstm")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CITIES = [
    "chennai",
    "delhi",
]

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(
    f"Using device: {DEVICE}"
)


# ============================================================
# MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout
            if num_layers > 1
            else 0.0,

        )

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )

        )


    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(
            last_output
        )

        return prediction.squeeze(
            -1
        )


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
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    batch_size=128,
    learning_rate=0.001
):

    model = model.to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # --------------------------------------------------------
    # Convert arrays to tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    n_samples = len(
        X_train_tensor
    )


    for epoch in range(
        epochs
    ):

        model.train()

        permutation = torch.randperm(
            n_samples
        )


        epoch_loss = 0.0


        for start in range(
            0,
            n_samples,
            batch_size
        ):

            indices = permutation[
                start:
                start + batch_size
            ]


            batch_X = (
                X_train_tensor[
                    indices
                ].to(DEVICE)
            )

            batch_y = (
                y_train_tensor[
                    indices
                ].to(DEVICE)
            )


            optimizer.zero_grad()


            predictions = model(
                batch_X
            )


            loss = criterion(
                predictions,
                batch_y
            )


            loss.backward()


            optimizer.step()


            epoch_loss += (
                loss.item()
                *
                len(indices)
            )


        epoch_loss /= n_samples


        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_predictions = model(
                X_val_tensor.to(DEVICE)
            )

            val_loss = criterion(
                val_predictions,
                y_val_tensor.to(DEVICE)
            ).item()


        if (
            epoch == 0
            or
            (epoch + 1) % 5 == 0
        ):

            print(
                f"Epoch "
                f"{epoch + 1:02d}/{epochs} "
                f"| Train Loss: "
                f"{epoch_loss:.6f} "
                f"| Val Loss: "
                f"{val_loss:.6f}"
            )


    return model


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    model,
    X
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)


    with torch.no_grad():

        predictions = model(
            X_tensor
        ).cpu().numpy()


    return predictions


# ============================================================
# MAIN LOOP
# ============================================================

results = []


for city in CITIES:

    print()
    print("=" * 70)
    print(
        f"LSTM TRAINING: {city.upper()}"
    )
    print("=" * 70)


    city_dir = (
        INPUT_DIR / city
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train = np.load(
        city_dir / "X_train_nn.npy"
    )

    y_train = np.load(
        city_dir / "y_train_nn.npy"
    )

    X_internal = np.load(
        city_dir / "X_internal_val_nn.npy"
    )

    y_internal = np.load(
        city_dir / "y_internal_val_nn.npy"
    )

    X_oot = np.load(
        city_dir / "X_oot_nn.npy"
    )

    y_oot = np.load(
        city_dir / "y_oot_nn.npy"
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

    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )


    print()
    print("Training LSTM...")


    model = train_model(

        model,

        X_train,
        y_train,

        X_internal,
        y_internal,

        epochs=30,

        batch_size=128,

        learning_rate=0.001,

    )


    # ========================================================
    # INTERNAL VALIDATION
    # ========================================================

    internal_pred_scaled = predict(
        model,
        X_internal
    )


    # ========================================================
    # OOT
    # ========================================================

    oot_pred_scaled = predict(
        model,
        X_oot
    )


    # --------------------------------------------------------
    # Load target scaler
    # --------------------------------------------------------

    import pickle

    with open(
        INPUT_DIR /
        "scalers" /
        f"{city}_target_scaler.pkl",
        "rb"
    ) as f:

        target_scaler = pickle.load(
            f
        )


    # --------------------------------------------------------
    # Convert predictions back to MW
    # --------------------------------------------------------

    internal_pred = (
        target_scaler
        .inverse_transform(
            internal_pred_scaled.reshape(
                -1,
                1
            )
        )
        .ravel()
    )


    oot_pred = (
        target_scaler
        .inverse_transform(
            oot_pred_scaled.reshape(
                -1,
                1
            )
        )
        .ravel()
    )


    # --------------------------------------------------------
    # Actual values
    # --------------------------------------------------------

    internal_actual = (
        target_scaler
        .inverse_transform(
            y_internal.reshape(
                -1,
                1
            )
        )
        .ravel()
    )


    oot_actual = (
        target_scaler
        .inverse_transform(
            y_oot.reshape(
                -1,
                1
            )
        )
        .ravel()
    )


    # ========================================================
    # METRICS
    # ========================================================

    internal_metrics = calculate_metrics(
        internal_actual,
        internal_pred
    )

    oot_metrics = calculate_metrics(
        oot_actual,
        oot_pred
    )


    # --------------------------------------------------------
    # Print results
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

    torch.save(

        model.state_dict(),

        OUTPUT_DIR /
        f"{city}_lstm.pt"

    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    # ========================================================
    # SAVE INTERNAL VALIDATION PREDICTIONS
    # ========================================================

    internal_prediction_df = pd.DataFrame({

        "Actual_Power_MW":
            internal_actual,

        "Predicted_Power_MW":
            internal_pred,

    })

    internal_prediction_df.to_csv(

        OUTPUT_DIR /
        f"{city}_internal_predictions.csv",

        index=False

    )


    # ========================================================
    # SAVE 2025 OOT PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({

        "Actual_Power_MW":
            oot_actual,

        "Predicted_Power_MW":
            oot_pred,

    })

    prediction_df.to_csv(

        OUTPUT_DIR /
        f"{city}_oot_predictions.csv",

        index=False

    )


    results.append({

        "City": city,

        "Dataset": "2025 OOT",

        **oot_metrics,

    })

# ============================================================
# SUMMARY
# ============================================================

import pandas as pd


results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_DIR /
    "lstm_results.csv",
    index=False
)


print()
print("=" * 70)
print("LSTM TRAINING COMPLETE")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False
    )
)