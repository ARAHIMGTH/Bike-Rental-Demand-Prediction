import joblib
import pandas as pd

from pathlib import Path


# --------------------------------
# MODEL PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"

INTERVAL_PATH = (
    BASE_DIR / "models" /
    "prediction_interval_radius.pkl"
)


# --------------------------------
# LOAD MODEL
# --------------------------------

model = joblib.load(MODEL_PATH)

feature_names = joblib.load(FEATURE_PATH)

interval_radius = float(
    joblib.load(INTERVAL_PATH)
)


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------

def predict_demand(data):

    hour = int(data["hour"])

    day_of_week = int(data["day_of_week"])

    season = data["season"]

    functioning_day = int(data["functioning_day"])

    input_data = {

        "Hour": hour,

        "Temperature": float(data["temperature"]),

        "Humidity": float(data["humidity"]),

        "Wind_Speed": float(data["wind_speed"]),

        "Visibility": float(data["visibility"]),

        "Dew_Point": float(data["dew_point"]),

        "Solar_Radiation": float(data["solar_radiation"]),

        "Rainfall": float(data["rainfall"]),

        "Snowfall": float(data["snowfall"]),

        "IsHoliday": int(data["holiday"]),

        "IsFunctioningDay": functioning_day,

        "Month": int(data["month"]),

        "Day": int(data["day"]),

        "DayOfWeek": day_of_week,

        "IsWeekend": int(day_of_week >= 5),

        "IsRushHour": int(
            hour in [7, 8, 9, 17, 18, 19]
        ),

        "Season_Autumn": int(season == "Autumn"),

        "Season_Spring": int(season == "Spring"),

        "Season_Summer": int(season == "Summer"),

        "Season_Winter": int(season == "Winter")

    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Non-functioning day means zero demand

    if functioning_day == 0:

        return {
            "minimum": 0,
            "expected": 0,
            "maximum": 0
        }

    prediction = float(
        model.predict(input_df)[0]
    )

    expected = max(
        0,
        round(prediction)
    )

    minimum = max(
        0,
        round(prediction - interval_radius)
    )

    maximum = max(
        expected,
        round(prediction + interval_radius)
    )

    return {

        "minimum": minimum,

        "expected": expected,

        "maximum": maximum

    }