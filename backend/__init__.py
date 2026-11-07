import joblib
import pandas as pd

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"


# Load trained model
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found!")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError("Feature names file not found!")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    return model, list(features)


model, feature_names = load_model()


# Prediction function
def predict_demand(data):

    season = data["season"]

    input_data = {

        "Hour": int(data["hour"]),
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
        "DayOfWeek": int(data["day_of_week"]),

        "IsWeekend": int(data["day_of_week"] >= 5),

        "IsRushHour": int(
            int(data["hour"]) in [7, 8, 9, 17, 18, 19]
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

    if not int(data["functioning_day"]):
        prediction = 0
    else:
        prediction = float(model.predict(input_df)[0])

    return max(0, round(prediction))
