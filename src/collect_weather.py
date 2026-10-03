import openmeteo_requests
import pandas as pd
import requests_cache

from retry_requests import retry


# ============================================================
# 1. Open-Meteo API setup
# ============================================================

URL = "https://archive-api.open-meteo.com/v1/archive"

cache_session = requests_cache.CachedSession(
    ".cache",
    expire_after=-1
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)


# ============================================================
# 2. Locations
# ============================================================

LOCATIONS = {

    "Chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707
    },

    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090
    }
}


# ============================================================
# 3. Years
# ============================================================

YEARS = [2024, 2025]


# ============================================================
# 4. Weather variables
# ============================================================

HOURLY_VARIABLES = [

    "shortwave_radiation",
    "direct_normal_irradiance",

    "temperature_2m",

    "cloud_cover",

    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",

    "surface_pressure",

    "relative_humidity_2m",

    "precipitation"
]


# ============================================================
# 5. Download data
# ============================================================

for year in YEARS:

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    for city, coordinates in LOCATIONS.items():

        print()
        print("=" * 60)
        print(f"Downloading {city} data for {year}")
        print("=" * 60)

        params = {

            "latitude":
                coordinates["latitude"],

            "longitude":
                coordinates["longitude"],

            "start_date":
                start_date,

            "end_date":
                end_date,

            "hourly":
                HOURLY_VARIABLES,

            "timezone":
                "Asia/Kolkata"
        }

        response = openmeteo.weather_api(
            URL,
            params=params
        )[0]

        hourly = response.Hourly()


        # ----------------------------------------------------
        # Create hourly timestamps
        # ----------------------------------------------------

        timestamps = pd.date_range(

            start=pd.to_datetime(
                hourly.Time(),
                unit="s",
                utc=True
            ).tz_convert("Asia/Kolkata"),

            end=pd.to_datetime(
                hourly.TimeEnd(),
                unit="s",
                utc=True
            ).tz_convert("Asia/Kolkata"),

            freq="h",

            inclusive="left"
        )


        # ----------------------------------------------------
        # Create dataframe
        # ----------------------------------------------------

        df = pd.DataFrame({

            "Timestamp":
                timestamps,

            "GHI":
                hourly.Variables(0).ValuesAsNumpy(),

            "DNI":
                hourly.Variables(1).ValuesAsNumpy(),

            "Temperature":
                hourly.Variables(2).ValuesAsNumpy(),

            "Cloud_Cover":
                hourly.Variables(3).ValuesAsNumpy(),

            "Wind_Speed":
                hourly.Variables(4).ValuesAsNumpy(),

            "Wind_Direction":
                hourly.Variables(5).ValuesAsNumpy(),

            "Wind_Gusts":
                hourly.Variables(6).ValuesAsNumpy(),

            "Pressure":
                hourly.Variables(7).ValuesAsNumpy(),

            "Humidity":
                hourly.Variables(8).ValuesAsNumpy(),

            "Precipitation":
                hourly.Variables(9).ValuesAsNumpy()
        })


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        filename = (
            f"data/raw/"
            f"{city.lower()}_{year}_weather.csv"
        )

        df.to_csv(
            filename,
            index=False
        )


        # ----------------------------------------------------
        # Basic verification
        # ----------------------------------------------------

        print(f"Saved: {filename}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(
            f"Missing values: "
            f"{df.isna().sum().sum()}"
        )

print()
print("=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)