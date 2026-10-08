import json
import pandas as pd
from pathlib import Path

# 1 km² site
TOTAL_AREA_KM2 = 1.0

SOLAR_LAND_FRACTION = 0.60
WIND_LAND_FRACTION = 0.40

SOLAR_CAPACITY_MW = 54.0
WIND_CAPACITY_MW = 4.0

# Processed data
solar_file = "data/processed/chennai_2025_power.csv"

df = pd.read_csv(solar_file)

# Hourly power (MW) -> energy (MWh)
annual_solar_generation_mwh = df["Solar_Power_MW"].sum()
annual_wind_generation_mwh = df["Wind_Power_MW"].sum()
annual_total_generation_mwh = (
    annual_solar_generation_mwh + annual_wind_generation_mwh
)

# Areas
solar_area_km2 = TOTAL_AREA_KM2 * SOLAR_LAND_FRACTION
wind_area_km2 = TOTAL_AREA_KM2 * WIND_LAND_FRACTION

# Capacity
hybrid_capacity_mw = SOLAR_CAPACITY_MW + WIND_CAPACITY_MW

# Financial parameters
ELECTRICITY_PRICE_INR_PER_KWH = 5.0
SOLAR_CAPEX_INR_PER_KW = 45000
WIND_CAPEX_INR_PER_KW = 80000
ANNUAL_OM_FRACTION = 0.02
PROJECT_LIFETIME_YEARS = 25

# CAPEX
solar_capex_inr = SOLAR_CAPACITY_MW * 1000 * SOLAR_CAPEX_INR_PER_KW
wind_capex_inr = WIND_CAPACITY_MW * 1000 * WIND_CAPEX_INR_PER_KW
total_capex_inr = solar_capex_inr + wind_capex_inr

# Annual revenue
annual_revenue_inr = (
    annual_total_generation_mwh
    * 1000
    * ELECTRICITY_PRICE_INR_PER_KWH
)

# Annual O&M
annual_om_inr = total_capex_inr * ANNUAL_OM_FRACTION

# Annual net revenue
annual_net_revenue_inr = annual_revenue_inr - annual_om_inr

# Payback period
payback_years = total_capex_inr / annual_net_revenue_inr

# 25-year ROI
lifetime_net_revenue_inr = (
    annual_net_revenue_inr * PROJECT_LIFETIME_YEARS
)

roi_25_year_percent = (
    (lifetime_net_revenue_inr - total_capex_inr)
    / total_capex_inr
) * 100

stats = {
    "land_area_km2": TOTAL_AREA_KM2,
    "solar_area_km2": solar_area_km2,
    "wind_area_km2": wind_area_km2,
    "solar_capacity_mw": SOLAR_CAPACITY_MW,
    "wind_capacity_mw": WIND_CAPACITY_MW,
    "hybrid_capacity_mw": hybrid_capacity_mw,
    "annual_solar_generation_mwh": annual_solar_generation_mwh,
    "annual_wind_generation_mwh": annual_wind_generation_mwh,
    "annual_total_generation_mwh": annual_total_generation_mwh,
    "solar_capex_inr": solar_capex_inr,
    "wind_capex_inr": wind_capex_inr,
    "total_capex_inr": total_capex_inr,
    "annual_revenue_inr": annual_revenue_inr,
    "annual_om_inr": annual_om_inr,
    "annual_net_revenue_inr": annual_net_revenue_inr,
    "payback_years": payback_years,
    "roi_25_year_percent": roi_25_year_percent
}

output_dir = Path("results/dashboard")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "chennai.json"

with open(output_file, "w") as f:
    json.dump(stats, f, indent=4)

print(f"Saved to: {output_file}")