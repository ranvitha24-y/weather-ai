"""Train and save five date-based weather prediction models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# Use only the new six-month dataset. Keep paths relative to this script so
# training works no matter which directory the command is run from.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "weather.csv"

# These names and this order must match predict.py and app.py exactly.
DATE_FEATURES = ["Year", "Month", "Day", "DayOfYear", "DayOfWeek"]
NUMERIC_TARGETS = {
    "temperature": BASE_DIR / "temperature_model.pkl",
    "humidity": BASE_DIR / "humidity_model.pkl",
    "pressure": BASE_DIR / "pressure_model.pkl",
    "rainfall": BASE_DIR / "rainfall_model.pkl",
}
WEATHER_MODEL_FILE = BASE_DIR / "weather_model.pkl"
ALLOWED_WEATHER_LABELS = {"Sunny", "Cloudy", "Rainy"}

# These settings limit overfitting while keeping the trees simple enough for
# this small daily dataset.
TREE_SETTINGS = {
    "max_depth": 6,
    "min_samples_leaf": 2,
    "random_state": 42,
}


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
    usable_rows = target.notna() & np.isfinite(target)
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

    model = DecisionTreeRegressor(**TREE_SETTINGS)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)

    # Refit on every usable row before saving the production model.
    model.fit(X_target, y_target)
    joblib.dump(model, model_file)
    print(
        f"Trained {target_name} model on {len(y_target)} rows "
        f"(test MAE: {error:.2f})."
    )
    print(f"Saved {model_file.name}")


def train_weather_model(X, target):
    """Train and save the Decision Tree classification model."""
    # Only exact supported labels are usable; invalid labels are not renamed.
    usable_rows = target.notna() & target.isin(ALLOWED_WEATHER_LABELS)
    X_target = X.loc[usable_rows]
    y_target = target.loc[usable_rows]

    invalid_label_count = int((~usable_rows).sum())
    if invalid_label_count:
        print(
            f"weather: skipped {invalid_label_count} row(s) with "
            "missing or invalid labels."
        )

    if y_target.nunique() < 2:
        raise ValueError("Weather must contain at least two different classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_target,
        y_target,
        test_size=0.20,
        random_state=42,
        stratify=y_target,
    )

    model = DecisionTreeClassifier(**TREE_SETTINGS)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Refit on every usable row before saving the production model.
    model.fit(X_target, y_target)
    joblib.dump(model, WEATHER_MODEL_FILE)
    print(
        f"Trained weather model on {len(y_target)} rows "
        f"(test accuracy: {accuracy:.2%})."
    )
    print(f"Saved {WEATHER_MODEL_FILE.name}")


def main():
    """Load the CSV and train all five models."""
    print(f"Dataset file: {DATA_FILE.name}")
    data = pd.read_csv(DATA_FILE)
    print(f"Rows loaded: {len(data)}")

    # Validate the new six-month CSV schema before processing any values.
    required_columns = ["date"] + list(NUMERIC_TARGETS) + ["weather"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {', '.join(missing_columns)}"
        )

    # Parse DD-MM-YYYY strictly. Rows with missing or invalid dates cannot
    # produce meaningful date features, so they are safely excluded.
    parsed_dates = pd.to_datetime(
        data["date"], format="%d-%m-%Y", errors="coerce"
    )
    valid_date_rows = parsed_dates.notna()
    if valid_date_rows.sum() < 2:
        raise ValueError("The CSV needs at least two valid dates.")

    invalid_date_count = int((~valid_date_rows).sum())
    if invalid_date_count:
        print(f"Skipped {invalid_date_count} row(s) with invalid dates.")

    data = data.loc[valid_date_rows].copy()
    parsed_dates = parsed_dates.loc[valid_date_rows]
    X = create_date_features(parsed_dates)[DATE_FEATURES]

    print(f"Rows with valid dates: {len(data)}")
    print(f"Feature columns: {', '.join(DATE_FEATURES)}")
    print(f"Target columns: {', '.join(list(NUMERIC_TARGETS) + ['weather'])}")

    # Invalid or missing numeric values are skipped only for their target;
    # no values are filled, generated, duplicated, or otherwise invented.
    for target_name, model_file in NUMERIC_TARGETS.items():
        numeric_target = pd.to_numeric(data[target_name], errors="coerce")
        invalid_target_count = int(
            (numeric_target.isna() | ~np.isfinite(numeric_target)).sum()
        )
        if invalid_target_count:
            print(
                f"{target_name}: skipped {invalid_target_count} row(s) "
                "with missing or invalid values."
            )
        train_numeric_model(
            X, numeric_target, target_name, model_file
        )

    train_weather_model(X, data["weather"])
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
