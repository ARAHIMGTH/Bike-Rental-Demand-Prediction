from flask import Flask, render_template, request, jsonify
from backend.predictor import predict_demand

app = Flask(__name__)


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------
# PREDICTION API
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Receive input from frontend
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid JSON data received."
            }), 400

        # Required input fields
        required_fields = [
            "hour",
            "temperature",
            "humidity",
            "wind_speed",
            "visibility",
            "dew_point",
            "solar_radiation",
            "rainfall",
            "snowfall",
            "holiday",
            "functioning_day",
            "month",
            "day",
            "day_of_week",
            "season"
        ]

        # Check missing fields
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": "Some required fields are missing.",
                "missing_fields": missing_fields
            }), 400

        # Generate prediction
        result = predict_demand(data)

        # Send prediction to frontend
        return jsonify({
            "success": True,
            "predicted_demand": result,
            "message": "Prediction generated successfully."
        })

    except (ValueError, TypeError, KeyError) as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception:

        app.logger.exception("Prediction error")

        return jsonify({
            "success": False,
            "error": "Something went wrong while generating prediction."
        }), 500


# --------------------------------
# RUN FLASK SERVER
# --------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )