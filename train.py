"""Train and save five date-based weather prediction models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# Keep file paths relative to this script so it works from any current directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "weather.csv"

DATE_FEATURES = ["Year", "Month", "Day", "DayOfYear", "DayOfWeek"]
NUMERIC_TARGETS = {
    "Temperature": BASE_DIR / "temperature_model.pkl",
    "Humidity": BASE_DIR / "humidity_model.pkl",
    "Pressure": BASE_DIR / "pressure_model.pkl",
    "Rainfall": BASE_DIR / "rainfall_model.pkl",
}
WEATHER_MODEL_FILE = BASE_DIR / "weather_model.pkl"


def create_date_features(dates):
    """Turn parsed dates into numeric values the models can use."""
    return pd.DataFrame(
        {
            "Year": dates.dt.year,
            "Month": dates.dt.month,
            "Day": dates.dt.day,
            "DayOfYear": dates.dt.dayofyear,
            "DayOfWeek": dates.dt.dayofweek,
        },
        index=dates.index,
    )


def train_numeric_model(X, target, target_name, model_file):
    """Train and save one Decision Tree regression model."""
    usable_rows = target.notna()
    X_target = X.loc[usable_rows]
    y_target = target.loc[usable_rows]

    if len(y_target) < 2:
        raise ValueError(f"{target_name} needs at least two usable rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_target,
        y_target,
        test_size=0.20,
        random_state=42,
    )

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)

    joblib.dump(model, model_file)
    print(f"{target_name} mean absolute error: {error:.2f}")
    print(f"Saved: {model_file.name}")


def train_weather_model(X, target):
    """Train and save the Decision Tree classification model."""
    usable_rows = target.notna() & target.astype(str).str.strip().ne("")
    X_target = X.loc[usable_rows]
    y_target = target.loc[usable_rows].astype(str).str.strip()

    if y_target.nunique() < 2:
        raise ValueError("Weather must contain at least two different classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_target,
        y_target,
        test_size=0.20,
        random_state=42,
        stratify=y_target,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, WEATHER_MODEL_FILE)
    print(f"Weather accuracy: {accuracy:.2%}")
    print(f"Saved: {WEATHER_MODEL_FILE.name}")


def main():
    """Load the CSV and train all five models."""
    data = pd.read_csv(DATA_FILE)

    required_columns = ["Date"] + list(NUMERIC_TARGETS) + ["Weather"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {', '.join(missing_columns)}"
        )

    parsed_dates = pd.to_datetime(
        data["Date"], format="%d-%m-%Y", errors="coerce"
    )
    valid_date_rows = parsed_dates.notna()
    if valid_date_rows.sum() < 2:
        raise ValueError("The CSV needs at least two valid dates.")

    data = data.loc[valid_date_rows].copy()
    parsed_dates = parsed_dates.loc[valid_date_rows]
    X = create_date_features(parsed_dates)[DATE_FEATURES]

    # Invalid numeric values become missing and are skipped only for that model.
    for target_name, model_file in NUMERIC_TARGETS.items():
        numeric_target = pd.to_numeric(data[target_name], errors="coerce")
        train_numeric_model(
            X, numeric_target, target_name, model_file
        )

    train_weather_model(X, data["Weather"])


if __name__ == "__main__":
    main()
