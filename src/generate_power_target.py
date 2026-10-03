import pandas as pd
import numpy as np
from pathlib import Path

from model_config import (
    PV_EFFICIENCY,
    PV_PERFORMANCE_RATIO,
    SOLAR_MODULE_AREA_M2,
    WIND_CAPACITY_MW,
    TURBINE_CUT_IN_SPEED,
    TURBINE_RATED_SPEED,
    TURBINE_CUT_OUT_SPEED,
    TURBINE_RATED_POWER_KW,
)


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FILES = [
    "chennai_2024_weather.csv",
    "chennai_2025_weather.csv",
    "delhi_2024_weather.csv",
    "delhi_2025_weather.csv",
]


# ============================================================
# SOLAR POWER MODEL
# ============================================================

# Installed PV capacity calculated from:
#
# Area × Standard Irradiance × Efficiency
#
# 1000 W/m² is the standard reference irradiance.

PV_CAPACITY_MW = (
    SOLAR_MODULE_AREA_M2
    * 1000
    * PV_EFFICIENCY
    / 1_000_000
)


def calculate_solar_power(ghi):
    """
    Calculate hourly solar power in MW.

    P = GHI × PV area × efficiency × performance ratio

    Output is capped at installed PV capacity.
    """

    power_mw = (
        ghi
        * SOLAR_MODULE_AREA_M2
        * PV_EFFICIENCY
        * PV_PERFORMANCE_RATIO
        / 1_000_000
    )

    power_mw = np.clip(
        power_mw,
        0,
        PV_CAPACITY_MW
    )

    return power_mw


# ============================================================
# WIND POWER MODEL
# ============================================================

def calculate_wind_power(wind_speed):
    """
    Simplified turbine power curve.

    Below cut-in:
        0 MW

    Between cut-in and rated speed:
        cubic interpolation

    Between rated and cut-out:
        rated power

    At/above cut-out:
        0 MW
    """

    v = np.asarray(
        wind_speed,
        dtype=float
    )

    power = np.zeros_like(v)

    # --------------------------------------------------------
    # Region 1: cut-in → rated
    # --------------------------------------------------------

    mask_ramp = (
        (v >= TURBINE_CUT_IN_SPEED)
        &
        (v < TURBINE_RATED_SPEED)
    )

    power[mask_ramp] = (
        WIND_CAPACITY_MW
        *
        (
            (
                v[mask_ramp] ** 3
                -
                TURBINE_CUT_IN_SPEED ** 3
            )
            /
            (
                TURBINE_RATED_SPEED ** 3
                -
                TURBINE_CUT_IN_SPEED ** 3
            )
        )
    )


    # --------------------------------------------------------
    # Region 2: rated → cut-out
    # --------------------------------------------------------

    mask_rated = (
        (v >= TURBINE_RATED_SPEED)
        &
        (v < TURBINE_CUT_OUT_SPEED)
    )

    power[mask_rated] = WIND_CAPACITY_MW


    # --------------------------------------------------------
    # Region 3: cut-out and above
    # --------------------------------------------------------

    mask_cutout = (
        v >= TURBINE_CUT_OUT_SPEED
    )

    power[mask_cutout] = 0


    # Safety cap
    power = np.clip(
        power,
        0,
        WIND_CAPACITY_MW
    )

    return power


# ============================================================
# PROCESS DATASET
# ============================================================

for filename in FILES:

    input_path = RAW_DIR / filename

    print()
    print("=" * 70)
    print(f"PROCESSING: {filename}")
    print("=" * 70)

    df = pd.read_csv(
        input_path
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"]
    )


    # --------------------------------------------------------
    # Solar power
    # --------------------------------------------------------

    df["Solar_Power_MW"] = (
        calculate_solar_power(
            df["GHI"].values
        )
    )


    # --------------------------------------------------------
    # Wind power
    # --------------------------------------------------------

    df["Wind_Power_MW"] = (
        calculate_wind_power(
            df["Wind_Speed"].values
        )
    )


    # --------------------------------------------------------
    # Hybrid renewable generation
    # --------------------------------------------------------

    df["Actual_Power_MW"] = (
        df["Solar_Power_MW"]
        +
        df["Wind_Power_MW"]
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR /
        filename.replace(
            "_weather.csv",
            "_power.csv"
        )
    )

    df.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        f"Rows              : {len(df)}"
    )

    print(
        f"Solar capacity    : "
        f"{PV_CAPACITY_MW:.2f} MW"
    )

    print(
        f"Wind capacity     : "
        f"{WIND_CAPACITY_MW:.2f} MW"
    )

    print(
        f"Hybrid capacity   : "
        f"{PV_CAPACITY_MW + WIND_CAPACITY_MW:.2f} MW"
    )

    print()

    print(
        f"Solar mean        : "
        f"{df['Solar_Power_MW'].mean():.3f} MW"
    )

    print(
        f"Wind mean         : "
        f"{df['Wind_Power_MW'].mean():.3f} MW"
    )

    print(
        f"Hybrid mean       : "
        f"{df['Actual_Power_MW'].mean():.3f} MW"
    )

    print()

    print(
        f"Solar max         : "
        f"{df['Solar_Power_MW'].max():.3f} MW"
    )

    print(
        f"Wind max          : "
        f"{df['Wind_Power_MW'].max():.3f} MW"
    )

    print(
        f"Hybrid max        : "
        f"{df['Actual_Power_MW'].max():.3f} MW"
    )

    print()

    print(
        f"Saved: {output_path}"
    )


print()
print("=" * 70)
print("POWER TARGET GENERATION COMPLETE")
print("=" * 70)