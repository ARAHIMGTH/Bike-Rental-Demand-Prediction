import joblib
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"


# Load trained model and feature names
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("best_model.pkl not found!")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError("feature_names.pkl not found!")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, list(features)


model, feature_names = load_model()


# Predict bike rental demand
def predict_demand(data):

    hour = int(data["hour"])
    day_of_week = int(data["day_of_week"])
    season = data["season"]

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
        "IsFunctioningDay": int(data["functioning_day"]),
        "Month": int(data["month"]),
        "Day": int(data["day"]),
        "DayOfWeek": day_of_week,
        "IsWeekend": int(day_of_week >= 5),
        "IsRushHour": int(hour in [7, 8, 9, 17, 18, 19]),
        "Season_Autumn": int(season == "Autumn"),
        "Season_Spring": int(season == "Spring"),
        "Season_Summer": int(season == "Summer"),
        "Season_Winter": int(season == "Winter")
    }

    # Arrange columns according to trained model
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # If bikes are not operating, demand is zero
    if int(data["functioning_day"]) == 0:
        return 0

    prediction = model.predict(input_df)[0]

    return max(0, round(float(prediction)))