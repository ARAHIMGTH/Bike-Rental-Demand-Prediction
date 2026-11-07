#CONTRIBUTERS
@ FARHAN ZAFAR
@ MD KATIB 
@ MD SAMAR ALAM
@ ABDUR RAHIM 



#LIVE DEMO
https://bike-rental-demand-predict.streamlit.app/
# 🚴 RidePredict – Bike Rental Demand Prediction

### Machine Learning Based Bike Rental Demand Forecasting System

RidePredict is a machine learning-powered web application that predicts bike rental demand based on weather conditions, time, and calendar information.

The project uses an **Extra Trees Regressor** to estimate the number of bikes likely to be rented and provides minimum, expected, and maximum demand estimates through an interactive Streamlit dashboard.

---

## 🌐 Live Application

**Live Demo:** Coming Soon

**GitHub Repository:** [Bike Rental Demand Prediction](https://github.com/fzafar3103/Bike-Rental-Demand-Prediction)

---

## 📌 Project Overview

Bike rental demand changes depending on several factors, including temperature, humidity, rainfall, season, holidays, and time of day.

RidePredict uses historical bike rental data to identify these patterns and generate demand predictions.

The application provides an interactive interface where users can enter environmental and calendar conditions to estimate future bike rental demand.

---

## ✨ Key Features

- **Demand Prediction:** Predicts bike rental demand using machine learning.
- **Interactive Dashboard:** User-friendly Streamlit interface.
- **Weather Analysis:** Considers temperature, humidity, wind speed, visibility, rainfall, and snowfall.
- **Time-Based Prediction:** Uses hour, day, month, and day of the week.
- **Calendar Information:** Considers holidays, weekends, and functioning days.
- **Prediction Range:** Displays minimum, expected, and maximum demand estimates.
- **Model Evaluation:** Displays R² Score, MAE, and RMSE.
- **Actual vs Predicted Analysis:** Visualizes model predictions against actual test data.
- **Feature Importance:** Identifies the factors influencing bike rental demand.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Extra Trees Regressor | Prediction Model |
| Streamlit | Web Application |
| Plotly | Interactive Visualizations |
| Matplotlib | Data Visualization |
| Joblib | Model Serialization |
| Git & GitHub | Version Control |

---

## 🤖 Machine Learning Model

RidePredict uses the **Extra Trees Regressor (Extremely Randomized Trees)** algorithm.

The model learns relationships between historical rental demand and input features to predict bike rental counts.

### Model Configuration

- Algorithm: Extra Trees Regressor
- Number of Estimators: 50
- Maximum Depth: 12
- Minimum Samples Split: 5
- Minimum Samples Leaf: 3
- Maximum Features: 0.8
- Random State: 42

### Model Performance

| Evaluation Metric | Result |
|---|---:|
| R² Score | 73.75% |
| Mean Absolute Error (MAE) | 200.60 |
| Root Mean Squared Error (RMSE) | 289.80 |

*Note: These are the previously reported evaluation results. Actual performance may vary depending on the dataset and evaluation split.*

---

## 📊 Input Features

The model considers the following information:

**Time & Calendar**
- Hour
- Month
- Day
- Day of Week
- Weekend
- Rush Hour

**Weather Conditions**
- Temperature
- Humidity
- Wind Speed
- Visibility
- Dew Point
- Solar Radiation
- Rainfall
- Snowfall

**Additional Information**
- Season
- Holiday
- Functioning Day

---

## 📁 Project Structure

```text
Bike-Rental-Demand-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── backend/
│   ├── __init__.py
│   └── predictor.py
│
├── dataset/
│   └── train.csv
│
├── models/
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   ├── prediction_interval_radius.pkl
│   └── feature_importance.csv
│
├── static/
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── notebooks/
│
├── templates/
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/fzafar3103/Bike-Rental-Demand-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Bike-Rental-Demand-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

---

## 🚀 Deployment

RidePredict can be deployed using **Streamlit Community Cloud**.

1. Push the project repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the RidePredict repository.
5. Set the main file path to `app.py`.
6. Click Deploy.

Ensure that the required model files and dataset are available in the repository.

---

## 🔮 Future Improvements

- Real-time weather API integration.
- More advanced machine learning algorithms.
- Improved prediction accuracy through hyperparameter optimization.
- Interactive historical demand analysis.
- Location-based bike rental demand forecasting.
- Model retraining with updated rental datasets.

---

## 👨‍💻 Author

**Farhan Zafar**

Machine Learning | Python | Data Science | Web Development

---

## ⭐ Support

If you find this project interesting or useful, consider giving the repository a ⭐ on GitHub.

**RidePredict – Making Bike Rental Demand More Predictable.**
