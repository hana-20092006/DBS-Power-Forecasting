from model_config import *


print("=" * 60)
print("RENEWABLE ENERGY MODEL CONFIGURATION")
print("=" * 60)

print()

print(f"Total area       : {TOTAL_AREA_KM2} km²")
print(f"Total area       : {TOTAL_AREA_M2:,.0f} m²")

print()

print(
    f"Solar allocation : "
    f"{SOLAR_LAND_FRACTION * 100:.0f}%"
)

print(
    f"Solar area       : "
    f"{SOLAR_AREA_M2:,.0f} m²"
)

print()

print(
    f"Wind allocation  : "
    f"{WIND_LAND_FRACTION * 100:.0f}%"
)

print(
    f"Wind area        : "
    f"{WIND_AREA_M2:,.0f} m²"
)

print()

print(
    f"PV efficiency    : "
    f"{PV_EFFICIENCY * 100:.1f}%"
)

print(
    f"PV system losses : "
    f"{PV_SYSTEM_LOSS * 100:.1f}%"
)

print()

print(
    f"Turbine rating   : "
    f"{TURBINE_RATED_POWER_KW:,} kW"
)

print(
    f"Cut-in speed     : "
    f"{TURBINE_CUT_IN_SPEED} m/s"
)

print(
    f"Rated speed      : "
    f"{TURBINE_RATED_SPEED} m/s"
)

print(
    f"Cut-out speed    : "
    f"{TURBINE_CUT_OUT_SPEED} m/s"
)

print()

print(
    f"Electricity price: "
    f"₹{ELECTRICITY_PRICE_INR_PER_KWH}/kWh"
)

print(
    f"Project lifetime : "
    f"{PROJECT_LIFETIME_YEARS} years"
)

print()

print("=" * 60)
print("CONFIGURATION OK")
print("=" * 60)

# ------------------------------------------------------------
# SOLAR LAND-USE MODEL
# ------------------------------------------------------------

# Fraction of allocated solar land that can actually
# be occupied by PV module surface.

SOLAR_GROUND_COVERAGE_RATIO = 0.45

SOLAR_MODULE_AREA_M2 = (
    SOLAR_AREA_M2 *
    SOLAR_GROUND_COVERAGE_RATIO
)


# ------------------------------------------------------------
# WIND LAND-USE MODEL
# ------------------------------------------------------------

# Approximate land requirement per MW of wind capacity.
# This represents turbine spacing / site area rather than
# rotor swept area.

WIND_LAND_REQUIREMENT_KM2_PER_MW = 0.10

WIND_CAPACITY_MW = (
    WIND_AREA_M2 / 1_000_000
) / WIND_LAND_REQUIREMENT_KM2_PER_MW