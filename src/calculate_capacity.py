from model_config import *


print("=" * 65)
print("1 km² HYBRID RENEWABLE ENERGY CAPACITY")
print("=" * 65)

print()

# ------------------------------------------------------------
# Site
# ------------------------------------------------------------

print("SITE")
print("-" * 65)

print(
    f"Total site area        : "
    f"{TOTAL_AREA_KM2:.2f} km²"
)

print(
    f"Total site area        : "
    f"{TOTAL_AREA_M2:,.0f} m²"
)

print()


# ------------------------------------------------------------
# Solar
# ------------------------------------------------------------

print("SOLAR PV")
print("-" * 65)

print(
    f"Solar land allocation  : "
    f"{SOLAR_LAND_FRACTION * 100:.1f}%"
)

print(
    f"Solar land area        : "
    f"{SOLAR_AREA_M2:,.0f} m²"
)

print(
    f"Ground coverage ratio  : "
    f"{SOLAR_GROUND_COVERAGE_RATIO * 100:.1f}%"
)

print(
    f"PV module area         : "
    f"{SOLAR_MODULE_AREA_M2:,.0f} m²"
)


# PV electrical capacity:
#
# Area × irradiance × efficiency
#
# At standard test irradiance:
#
# 1000 W/m² × module area × efficiency

PV_CAPACITY_W = (
    SOLAR_MODULE_AREA_M2 *
    1000 *
    PV_EFFICIENCY
)

PV_CAPACITY_KW = (
    PV_CAPACITY_W / 1000
)

PV_CAPACITY_MW = (
    PV_CAPACITY_KW / 1000
)

print(
    f"Installed PV capacity : "
    f"{PV_CAPACITY_MW:.2f} MW"
)

print()


# ------------------------------------------------------------
# Wind
# ------------------------------------------------------------

print("WIND")
print("-" * 65)

print(
    f"Wind land allocation   : "
    f"{WIND_LAND_FRACTION * 100:.1f}%"
)

print(
    f"Wind land area         : "
    f"{WIND_AREA_M2:,.0f} m²"
)

print(
    f"Wind land requirement  : "
    f"{WIND_LAND_REQUIREMENT_KM2_PER_MW:.2f} km²/MW"
)

print(
    f"Equivalent wind        : "
    f"{WIND_CAPACITY_MW:.2f} MW"
)

print()


# ------------------------------------------------------------
# Hybrid capacity
# ------------------------------------------------------------

HYBRID_CAPACITY_MW = (
    PV_CAPACITY_MW +
    WIND_CAPACITY_MW
)

print("HYBRID SITE")
print("-" * 65)

print(
    f"Solar capacity         : "
    f"{PV_CAPACITY_MW:.2f} MW"
)

print(
    f"Wind capacity          : "
    f"{WIND_CAPACITY_MW:.2f} MW"
)

print(
    f"Hybrid capacity        : "
    f"{HYBRID_CAPACITY_MW:.2f} MW"
)

print()

print("=" * 65)
print("CAPACITY CALCULATION COMPLETE")
print("=" * 65)