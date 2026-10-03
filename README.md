# DBS Power Forecasting

A machine learning pipeline for **solar power forecasting** using weather and temporal features for **Chennai and Delhi**.

The project compares three forecasting approaches:

- **XGBoost**
- **LSTM**
- **Transformer**

It also includes validation, out-of-time (OOT) evaluation, ensemble prediction, SHAP-based explainability, and result visualizations.

## Project Overview

```text
Weather / Power Data
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Train / Internal Validation / 2025 OOT Split
        ↓
Model-Ready Data
        ↓
 ┌──────────┬──────────┬─────────────┐
 │ XGBoost  │   LSTM   │ Transformer │
 └──────────┴──────────┴─────────────┘
        ↓
Prediction & Evaluation
        ↓
Ensemble Weight Selection
        ↓
SHAP Explainability
        ↓
Visualizations
```

## Cities

The project currently evaluates:

- Chennai
- Delhi

The models are evaluated using a **2024 internal validation set** and a **2025 out-of-time (OOT) validation set**.

## Models

### XGBoost

Gradient-boosted decision trees are used as the tabular forecasting model with engineered weather, temporal, and lag features.

### LSTM

A Long Short-Term Memory neural network is used to capture temporal patterns from sequences. The neural-network sequence length is **24 time steps**.

### Transformer

A Transformer-based neural network is used to model temporal dependencies using attention mechanisms, also using 24-step sequences.

## Final 2025 OOT Results

| City | MAE (MW) | RMSE (MW) | R² | sMAPE |
|---|---:|---:|---:|---:|
| Chennai | 0.1824 | 0.3608 | 0.9993 | 15.02% |
| Delhi | 0.1678 | 0.3028 | 0.9994 | 8.04% |

### Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination
- **sMAPE** — Symmetric Mean Absolute Percentage Error

All final results above are from the **2025 out-of-time validation set**.

## Ensemble

An ensemble pipeline combines predictions from XGBoost, LSTM, and Transformer. Weights are selected using **2024 internal validation only**, without using the 2025 OOT set for weight selection.

Selected weights:

| City | XGBoost | LSTM | Transformer |
|---|---:|---:|---:|
| Chennai | 1.0 | 0.0 | 0.0 |
| Delhi | 1.0 | 0.0 | 0.0 |

Thus, the current final ensemble predictions correspond to the XGBoost predictions for both cities. The ensemble framework remains implemented for multi-model alignment and weighted prediction.

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to investigate which features have the largest contribution to XGBoost predictions.

### Chennai — Top Features

| Feature | Mean Absolute SHAP |
|---|---:|
| GHI | 9.1045 |
| Wind_Speed | 1.2057 |
| Actual_Power_MW_lag_24 | 1.1386 |
| GHI_lag_24 | 1.0684 |
| DNI | 0.5066 |

### Delhi — Top Features

| Feature | Mean Absolute SHAP |
|---|---:|
| GHI | 8.0916 |
| GHI_lag_24 | 1.5781 |
| Wind_Speed | 1.2040 |
| DNI | 0.7051 |
| Actual_Power_MW_lag_1 | 0.2017 |

## Project Structure

```text
DBS_Project/
│
├── data/
│   └── model_ready/
│       ├── chennai/
│       └── delhi/
│
├── models/
│   ├── ensemble/
│   ├── lstm/
│   ├── shap/
│   ├── transformer/
│   ├── visualizations/
│   └── xgboost/
│
├── results/
│   └── figures/
│
├── src/
│   ├── analyze_xgboost.py
│   ├── audit_features.py
│   ├── audit_predictions.py
│   ├── calculate_capacity.py
│   ├── collect_weather.py
│   ├── eda_raw_data.py
│   ├── ensemble.py
│   ├── feature_engineering.py
│   ├── generate_power_target.py
│   ├── investigate_wind.py
│   ├── model_config.py
│   ├── prepare_ml_data.py
│   ├── shap_analysis.py
│   ├── split_data.py
│   ├── test_config.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── train_xgboost.py
│   ├── validate_raw_data.py
│   └── visualize_results.py
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/hana-20092006/DBS-Power-Forecasting.git
cd DBS-Power-Forecasting
```

Create a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Pipeline

```powershell
python src/train_xgboost.py
python src/train_lstm.py
python src/train_transformer.py
python src/ensemble.py
python src/shap_analysis.py
python src/visualize_results.py
```

Generated outputs are stored under `models/` and `results/`.

## Visualizations

The project generates:

- Actual vs predicted power plots
- Model MAE comparison
- Model RMSE comparison
- Model R² comparison
- Ensemble MAE/RMSE/R²
- Prediction error distributions
- SHAP feature importance plots
- SHAP summary outputs

## Validation Strategy

### Training
Used to fit the models.

### 2024 Internal Validation
Used for model evaluation and ensemble weight selection.

### 2025 Out-of-Time Validation
Used as the final temporal holdout evaluation.

This provides a time-based test of generalization to a later period.

## Tech Stack

- Python
- XGBoost
- PyTorch
- scikit-learn
- SHAP
- NumPy
- Pandas
- Matplotlib
- Git / GitHub

## Key Takeaways

- Three forecasting architectures were implemented and evaluated.
- The project supports both tabular and sequential modeling approaches.
- Evaluation includes a dedicated **2025 out-of-time dataset**.
- An ensemble framework was implemented with validation-based weight selection.
- SHAP provides feature-level explainability for the XGBoost model.
- The final 2025 OOT results show high R² values for both Chennai and Delhi.

## Author

**Hana Maria Philip**

GitHub: [@hana-20092006](https://github.com/hana-20092006)

## License

This project is intended for academic and educational use.
