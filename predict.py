"""Predict weather measurements and condition from one date."""

from pathlib import Path
import sys

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATE_FEATURES = ["Year", "Month", "Day", "DayOfYear", "DayOfWeek"]
MODEL_FILES = {
    "Temperature": BASE_DIR / "temperature_model.pkl",
    "Humidity": BASE_DIR / "humidity_model.pkl",
    "Pressure": BASE_DIR / "pressure_model.pkl",
    "Rainfall": BASE_DIR / "rainfall_model.pkl",
    "Weather": BASE_DIR / "weather_model.pkl",
}


def main():
    """Create date features and make predictions with all five models."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    selected_date_text = input("Enter a date (DD-MM-YYYY): ").strip()

    selected_date = pd.to_datetime(
        selected_date_text, format="%d-%m-%Y", errors="coerce"
    )
    if pd.isna(selected_date):
        print("Invalid date. Please use DD-MM-YYYY format.")
        return

    date_values = pd.DataFrame(
        [
            {
                "Year": selected_date.year,
                "Month": selected_date.month,
                "Day": selected_date.day,
                "DayOfYear": selected_date.dayofyear,
                "DayOfWeek": selected_date.dayofweek,
            }
        ],
        columns=DATE_FEATURES,
    )

    predictions = {}
    for target_name, model_file in MODEL_FILES.items():
        model = joblib.load(model_file)
        predictions[target_name] = model.predict(date_values)[0]

    print(f"Temperature: {predictions['Temperature']:.1f} °C")
    print(f"Humidity: {predictions['Humidity']:.1f}%")
    print(f"Pressure: {predictions['Pressure']:.1f} hPa")
    print(f"Rainfall: {predictions['Rainfall']:.1f} mm")
    print(f"Weather: {predictions['Weather']}")


if __name__ == "__main__":
    main()
