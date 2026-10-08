from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

app = FastAPI(title="Renewable Energy Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "dashboard"


def load_json(filename):
    file_path = RESULTS_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found"
        )

    with open(file_path, "r") as f:
        return json.load(f)


@app.get("/")
def root():
    return {
        "message": "Renewable Energy Forecasting API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }

@app.get("/api/site-stats")
def site_stats(city: str = "chennai"):
    if city not in ["chennai", "delhi"]:
        raise HTTPException(status_code=400, detail="Invalid city")

    return load_json(f"{city}.json")

@app.get("/api/validation")
def validation():
    return load_json("validation.json")


@app.get("/api/models")
def models():
    return load_json("model_comparison.json")

@app.get("/api/forecast")
def forecast(city: str = "chennai"):
    file_path = (
        PROJECT_ROOT
        / "models"
        / "ensemble"
        / f"{city}_oot_ensemble_predictions.csv"
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Forecast data not found"
        )

    import pandas as pd

    df = pd.read_csv(file_path)

    # Keep the response small enough for the dashboard
    df = df.head(200)

    return df.to_dict(orient="records")

@app.get("/api/shap")
def shap_features(city: str = "chennai"):
    file_path = (
        PROJECT_ROOT
        / "models"
        / "shap"
        / f"{city}_shap_feature_importance.csv"
    )

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="SHAP data not found"
        )

    import pandas as pd

    df = pd.read_csv(file_path)

    # Show the 10 most important features
    df = df.head(10)

    return df.to_dict(orient="records")