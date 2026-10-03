import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/raw")
FIGURE_DIR = Path("results/figures")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = {

    "Chennai_2024":
        DATA_DIR / "chennai_2024_weather.csv",

    "Chennai_2025":
        DATA_DIR / "chennai_2025_weather.csv",

    "Delhi_2024":
        DATA_DIR / "delhi_2024_weather.csv",

    "Delhi_2025":
        DATA_DIR / "delhi_2025_weather.csv"
}


# ============================================================
# Load datasets
# ============================================================

data = {}

for name, path in DATASETS.items():

    df = pd.read_csv(path)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    data[name] = df

    print(
        f"Loaded {name}: "
        f"{len(df)} rows"
    )


# ============================================================
# 1. Summary statistics
# ============================================================

print()
print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

for name, df in data.items():

    print()
    print(f"--- {name} ---")

    columns = [
        "GHI",
        "DNI",
        "Temperature",
        "Cloud_Cover",
        "Wind_Speed",
        "Wind_Gusts",
        "Humidity",
        "Precipitation"
    ]

    print(
        df[columns]
        .describe()
        .round(2)
    )


# ============================================================
# 2. Annual averages
# ============================================================

print()
print("=" * 70)
print("ANNUAL AVERAGES")
print("=" * 70)

annual_rows = []

for name, df in data.items():

    row = {

        "Dataset": name,

        "GHI_mean":
            df["GHI"].mean(),

        "DNI_mean":
            df["DNI"].mean(),

        "Temperature_mean":
            df["Temperature"].mean(),

        "Cloud_Cover_mean":
            df["Cloud_Cover"].mean(),

        "Wind_Speed_mean":
            df["Wind_Speed"].mean(),

        "Humidity_mean":
            df["Humidity"].mean(),

        "Precipitation_mean":
            df["Precipitation"].mean()
    }

    annual_rows.append(row)


annual_summary = pd.DataFrame(
    annual_rows
)

print(
    annual_summary.round(2)
)


# ============================================================
# 3. Extreme wind observations
# ============================================================

print()
print("=" * 70)
print("EXTREME WIND SPEED OBSERVATIONS")
print("=" * 70)

for name, df in data.items():

    print()
    print(f"--- {name} ---")

    print(
        df["Wind_Speed"]
        .quantile(
            [0.90, 0.95, 0.99, 0.995, 1.00]
        )
        .round(2)
    )


# ============================================================
# 4. Very high wind observations
# ============================================================

print()
print("=" * 70)
print("WIND SPEED > 25 m/s")
print("=" * 70)

for name, df in data.items():

    count = (
        df["Wind_Speed"] > 25
    ).sum()

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{name:15s}: "
        f"{count:5d} rows "
        f"({percentage:.3f}%)"
    )


# ============================================================
# 5. Solar zero values
# ============================================================

print()
print("=" * 70)
print("ZERO GHI ANALYSIS")
print("=" * 70)

for name, df in data.items():

    zero_count = (
        df["GHI"] == 0
    ).sum()

    percentage = (
        zero_count / len(df) * 100
    )

    print(
        f"{name:15s}: "
        f"{zero_count:5d} rows "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 6. Monthly averages
# ============================================================

for name, df in data.items():

    df["Month"] = (
        df["Timestamp"].dt.month
    )

    monthly = (
        df.groupby("Month")[
            [
                "GHI",
                "DNI",
                "Temperature",
                "Wind_Speed",
                "Humidity"
            ]
        ]
        .mean()
    )

    print()
    print("=" * 70)
    print(f"MONTHLY AVERAGES — {name}")
    print("=" * 70)

    print(
        monthly.round(2)
    )


# ============================================================
# 7. Wind speed distributions
# ============================================================

plt.figure(figsize=(10, 6))

for name, df in data.items():

    plt.hist(
        df["Wind_Speed"],
        bins=50,
        alpha=0.35,
        label=name
    )

plt.xlabel("Wind Speed")
plt.ylabel("Frequency")
plt.title(
    "Wind Speed Distribution — Chennai vs Delhi"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "wind_speed_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. GHI distributions
# ============================================================

plt.figure(figsize=(10, 6))

for name, df in data.items():

    plt.hist(
        df["GHI"],
        bins=50,
        alpha=0.35,
        label=name
    )

plt.xlabel("GHI")
plt.ylabel("Frequency")
plt.title(
    "GHI Distribution — Chennai vs Delhi"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "ghi_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. Temperature distributions
# ============================================================

plt.figure(figsize=(10, 6))

for name, df in data.items():

    plt.hist(
        df["Temperature"],
        bins=50,
        alpha=0.35,
        label=name
    )

plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.title(
    "Temperature Distribution — Chennai vs Delhi"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "temperature_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. Correlation matrices
# ============================================================

correlation_columns = [
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

for name, df in data.items():

    correlation = (
        df[correlation_columns]
        .corr()
    )

    print()
    print("=" * 70)
    print(f"CORRELATION MATRIX — {name}")
    print("=" * 70)

    print(
        correlation.round(2)
    )


print()
print("=" * 70)
print("EDA COMPLETE")
print("=" * 70)