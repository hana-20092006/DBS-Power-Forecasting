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
# Investigation
# ============================================================

for filename in FILES:

    path = DATA_DIR / filename

    df = pd.read_csv(path)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )

    print()
    print("=" * 75)
    print(f"WIND INVESTIGATION — {filename}")
    print("=" * 75)


    # --------------------------------------------------------
    # 1. Wind speed distribution
    # --------------------------------------------------------

    print("\n1. Wind Speed Percentiles")
    print("-------------------------")

    percentiles = [
        0.50,
        0.75,
        0.90,
        0.95,
        0.975,
        0.99,
        0.995,
        0.999,
        1.00
    ]

    print(
        df["Wind_Speed"]
        .quantile(percentiles)
        .round(2)
    )


    # --------------------------------------------------------
    # 2. Count observations in wind-speed ranges
    # --------------------------------------------------------

    print("\n2. Wind Speed Ranges")
    print("--------------------")

    ranges = [
        ("< 3 m/s", df["Wind_Speed"] < 3),

        ("3–5 m/s",
         (df["Wind_Speed"] >= 3) &
         (df["Wind_Speed"] < 5)),

        ("5–10 m/s",
         (df["Wind_Speed"] >= 5) &
         (df["Wind_Speed"] < 10)),

        ("10–15 m/s",
         (df["Wind_Speed"] >= 10) &
         (df["Wind_Speed"] < 15)),

        ("15–20 m/s",
         (df["Wind_Speed"] >= 15) &
         (df["Wind_Speed"] < 20)),

        ("20–25 m/s",
         (df["Wind_Speed"] >= 20) &
         (df["Wind_Speed"] < 25)),

        ("25–30 m/s",
         (df["Wind_Speed"] >= 25) &
         (df["Wind_Speed"] < 30)),

        ("30–40 m/s",
         (df["Wind_Speed"] >= 30) &
         (df["Wind_Speed"] < 40)),

        ("40–50 m/s",
         (df["Wind_Speed"] >= 40) &
         (df["Wind_Speed"] < 50)),

        (">= 50 m/s",
         df["Wind_Speed"] >= 50)
    ]

    for label, mask in ranges:

        count = mask.sum()

        percentage = (
            count /
            len(df) *
            100
        )

        print(
            f"{label:12s}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)"
        )


    # --------------------------------------------------------
    # 3. Highest wind observations
    # --------------------------------------------------------

    print("\n3. Top 20 Wind-Speed Observations")
    print("----------------------------------")

    top_wind = (
        df[
            [
                "Timestamp",
                "Wind_Speed",
                "Wind_Gusts",
                "Temperature",
                "Pressure",
                "Humidity",
                "Precipitation"
            ]
        ]
        .sort_values(
            "Wind_Speed",
            ascending=False
        )
        .head(20)
    )

    print(
        top_wind
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # 4. Compare wind speed and gusts
    # --------------------------------------------------------

    print("\n4. Wind Speed vs Gusts")
    print("----------------------")

    print(
        df[
            [
                "Wind_Speed",
                "Wind_Gusts"
            ]
        ]
        .describe()
        .round(2)
    )


    # --------------------------------------------------------
    # 5. High-wind rows with context
    # --------------------------------------------------------

    print("\n5. Observations with Wind Speed >= 40 m/s")
    print("------------------------------------------")

    high_wind = (
        df[
            df["Wind_Speed"] >= 40
        ][
            [
                "Timestamp",
                "Wind_Speed",
                "Wind_Gusts",
                "Temperature",
                "Pressure",
                "Humidity",
                "Cloud_Cover",
                "Precipitation"
            ]
        ]
        .sort_values(
            "Wind_Speed",
            ascending=False
        )
    )

    print(
        high_wind
        .head(30)
        .to_string(index=False)
    )


print()
print("=" * 75)
print("WIND INVESTIGATION COMPLETE")
print("=" * 75)