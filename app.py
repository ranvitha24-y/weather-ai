"""Flask application setup for date-based weather prediction."""

import datetime
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load all trained models once when the application starts.
temperature_model = joblib.load(BASE_DIR / "temperature_model.pkl")
humidity_model = joblib.load(BASE_DIR / "humidity_model.pkl")
pressure_model = joblib.load(BASE_DIR / "pressure_model.pkl")
rainfall_model = joblib.load(BASE_DIR / "rainfall_model.pkl")
weather_model = joblib.load(BASE_DIR / "weather_model.pkl")


def forecast_for_date(date_text):
    """Validate a date and return predictions from all trained models."""
    try:
        selected_date = datetime.datetime.strptime(date_text, "%Y-%m-%d")
    except (TypeError, ValueError):
        raise ValueError("Please provide a valid date in YYYY-MM-DD format.")

    date_features = pd.DataFrame(
        [
            {
                "Year": selected_date.year,
                "Month": selected_date.month,
                "Day": selected_date.day,
                "DayOfYear": selected_date.timetuple().tm_yday,
                "DayOfWeek": selected_date.weekday(),
            }
        ]
    )

    return {
        "date": date_text,
        "temperature": round(float(temperature_model.predict(date_features)[0]), 1),
        "humidity": round(float(humidity_model.predict(date_features)[0]), 1),
        "pressure": round(float(pressure_model.predict(date_features)[0]), 1),
        "rainfall": round(float(rainfall_model.predict(date_features)[0]), 1),
        "weather": str(weather_model.predict(date_features)[0]),
    }


@app.after_request
def allow_lovable_frontend(response):
    """Allow the separately hosted Lovable frontend to call the public API."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True) or {}

    try:
        forecast = forecast_for_date(str(payload.get("date", "")).strip())
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(forecast)


@app.route("/predict", methods=["POST"])
def predict():
    date_text = request.form.get("date", "").strip()

    try:
        forecast = forecast_for_date(date_text)
    except ValueError as error:
        return render_template(
            "index.html",
            selected_date=date_text,
            error=str(error),
        )

    return render_template(
        "index.html",
        selected_date=date_text,
        temperature=forecast["temperature"],
        humidity=forecast["humidity"],
        pressure=forecast["pressure"],
        rainfall=forecast["rainfall"],
        weather=forecast["weather"],
    )


if __name__ == "__main__":
    # Listen on the local network so other devices on the same Wi-Fi can connect.
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
