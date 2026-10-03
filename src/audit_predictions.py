import os
import pandas as pd

BASE = "models"

CITIES = ["chennai", "delhi"]

MODEL_FILES = {
    "xgboost": {
        "oot": "{city}_oot_predictions.csv",
        "internal": "{city}_internal_predictions.csv",
    },
    "lstm": {
        "oot": "{city}_oot_predictions.csv",
        "internal": "{city}_internal_predictions.csv",
    },
    "transformer": {
        "oot": "{city}_oot_predictions.csv",
        "internal": "{city}_internal_predictions.csv",
    },
}


def inspect_file(path):
    print(f"\nFile: {path}")

    if not os.path.exists(path):
        print("  STATUS: MISSING")
        return None

    df = pd.read_csv(path)

    print(f"  Rows    : {len(df)}")
    print(f"  Columns : {list(df.columns)}")

    print("\n  First 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\n  Missing values:")
    print(df.isna().sum().to_dict())

    return df


print("=" * 70)
print("PREDICTION FILE AUDIT")
print("=" * 70)

for city in CITIES:

    print("\n" + "=" * 70)
    print(f"CITY: {city.upper()}")
    print("=" * 70)

    for model, files in MODEL_FILES.items():

        print("\n" + "-" * 70)
        print(f"MODEL: {model.upper()}")
        print("-" * 70)

        for dataset_type, filename in files.items():

            path = os.path.join(
                BASE,
                model,
                filename.format(city=city)
            )

            inspect_file(path)


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)