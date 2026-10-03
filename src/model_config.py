# ============================================================
# Renewable Energy Model Configuration
# ============================================================

# ------------------------------------------------------------
# LAND
# ------------------------------------------------------------

TOTAL_AREA_KM2 = 1.0

TOTAL_AREA_M2 = (
    TOTAL_AREA_KM2 * 1_000_000
)


# ------------------------------------------------------------
# LAND ALLOCATION
# ------------------------------------------------------------

# Fraction of the 1 km² site allocated to each technology.

SOLAR_LAND_FRACTION = 0.60
WIND_LAND_FRACTION = 0.40


SOLAR_AREA_M2 = (
    TOTAL_AREA_M2 *
    SOLAR_LAND_FRACTION
)

WIND_AREA_M2 = (
    TOTAL_AREA_M2 *
    WIND_LAND_FRACTION
)


# ------------------------------------------------------------
# SOLAR PV
# ------------------------------------------------------------

PV_EFFICIENCY = 0.20

PV_SYSTEM_LOSS = 0.15

PV_PERFORMANCE_RATIO = (
    1 - PV_SYSTEM_LOSS
)


# ------------------------------------------------------------
# WIND TURBINE
# ------------------------------------------------------------

# Representative utility-scale turbine
# used for the physics-based model.

TURBINE_RATED_POWER_KW = 3000

TURBINE_CUT_IN_SPEED = 3.0

TURBINE_RATED_SPEED = 12.0

TURBINE_CUT_OUT_SPEED = 25.0

TURBINE_ROTOR_DIAMETER_M = 130.0


# ------------------------------------------------------------
# AIR DENSITY
# ------------------------------------------------------------

AIR_DENSITY_KG_M3 = 1.225


# ------------------------------------------------------------
# WIND POWER COEFFICIENT
# ------------------------------------------------------------

POWER_COEFFICIENT = 0.40


# ------------------------------------------------------------
# ECONOMIC PARAMETERS
# ------------------------------------------------------------

ELECTRICITY_PRICE_INR_PER_KWH = 5.0

PROJECT_LIFETIME_YEARS = 25

ANNUAL_OM_FRACTION = 0.02


# ------------------------------------------------------------
# CAPEX ASSUMPTIONS
# ------------------------------------------------------------

SOLAR_CAPEX_INR_PER_KW = 45_000

WIND_CAPEX_INR_PER_KW = 80_000

# ------------------------------------------------------------
# SOLAR LAND-USE MODEL
# ------------------------------------------------------------

SOLAR_GROUND_COVERAGE_RATIO = 0.45

SOLAR_MODULE_AREA_M2 = (
    SOLAR_AREA_M2 *
    SOLAR_GROUND_COVERAGE_RATIO
)


# ------------------------------------------------------------
# WIND LAND-USE MODEL
# ------------------------------------------------------------

WIND_LAND_REQUIREMENT_KM2_PER_MW = 0.10

WIND_CAPACITY_MW = (
    WIND_AREA_M2 / 1_000_000
) / WIND_LAND_REQUIREMENT_KM2_PER_MW