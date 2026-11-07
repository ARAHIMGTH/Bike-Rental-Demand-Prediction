import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RidePredict | Bike Rental Demand",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
INTERVAL_PATH = BASE_DIR / "models" / "prediction_interval_radius.pkl"
DATA_PATH = BASE_DIR / "dataset" / "train.csv"
IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"


# =====================================================
# COLOUR PALETTE
# =====================================================

BG = "#05161A"
SIDEBAR = "#072E33"
TEAL = "#0C7075"
ACCENT = "#0F969C"
BLUE = "#6DA5C0"
SLATE = "#294D61"

TEXT = "#EAF4F4"
MUTED = "#A8C0C4"
CARD = "#0A252B"
INPUT = "#0B3037"
BORDER = "rgba(109,165,192,0.22)"


# =====================================================
# CSS
# =====================================================

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: Inter, sans-serif;
}}

.stApp {{
    background: radial-gradient(
        ellipse at 80% 0%,
        rgba(15,150,156,.10),
        transparent 35%
    ), {BG};
    color: {TEXT};
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SIDEBAR}, {BG});
    border-right: 1px solid {BORDER};
}}

.block-container {{
    max-width: 1750px;
    padding: 1.7rem 2rem 2.5rem;
}}

h1, h2, h3, h4, p, label {{
    color: {TEXT};
}}

.brand {{
    text-align: center;
    padding: 10px 0 24px;
}}

.brand-icon {{
    font-size: 42px;
}}

.brand-name {{
    font-size: 27px;
    font-weight: 800;
}}

.brand-name span {{
    color: #28C5C0;
}}

.brand-caption {{
    color: {MUTED};
    font-size: 10px;
    letter-spacing: 1.2px;
}}

.main-title {{
    font-size: clamp(28px, 2.4vw, 40px);
    font-weight: 800;
    letter-spacing: -1px;
}}

.title-accent {{
    color: #28C5C0;
}}

.subtitle {{
    color: {MUTED};
    font-size: 14px;
    margin-top: 6px;
}}

.panel {{
    background: linear-gradient(145deg, {CARD}, {SIDEBAR});
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 20px;
    box-shadow:
        7px 8px 20px rgba(0,0,0,.18),
        inset 1px 1px 0 rgba(255,255,255,.025);
    margin-bottom: 16px;
}}

.panel-title {{
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 15px;
}}

.section-label {{
    color: #7FD6D4;
    font-weight: 700;
    font-size: 13px;
    padding: 9px 12px;
    margin: 12px 0;
    border-radius: 9px;
    background: rgba(15,150,156,.13);
}}

.metric-card {{
    background: linear-gradient(145deg, {INPUT}, {CARD});
    border: 1px solid {BORDER};
    border-radius: 15px;
    padding: 18px 10px;
    min-height: 105px;
    text-align: center;
    box-shadow: 6px 7px 17px rgba(0,0,0,.15);
    transition: .2s ease;
}}

.metric-card:hover {{
    transform: translateY(-3px);
    border-color: {ACCENT};
}}

.metric-label {{
    color: {MUTED};
    font-size: 11px;
    font-weight: 700;
}}

.metric-value {{
    color: #35C8C2;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}}

.expected {{
    color: {BLUE};
}}

.maximum {{
    color: #8BC8D7;
}}

.note {{
    background: rgba(12,112,117,.15);
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 10px;
    padding: 13px;
    color: {MUTED};
    font-size: 12px;
}}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background: {INPUT};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

.stButton > button,
.stFormSubmitButton > button {{
    width: 100%;
    min-height: 48px;
    border-radius: 11px;
    color: white;
    font-weight: 750;
    border: 1px solid rgba(109,165,192,.4);
    background: linear-gradient(120deg, {TEAL}, {ACCENT});
    box-shadow:
        0 6px 18px rgba(15,150,156,.2),
        inset 0 1px 1px rgba(255,255,255,.2);
    transition: .2s ease;
}}

.stButton > button:hover,
.stFormSubmitButton > button:hover {{
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 9px 22px rgba(15,150,156,.3);
}}

div[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 13px;
}}

hr {{
    border-color: {BORDER};
}}

footer {{
    visibility: hidden;
}}

#MainMenu {{
    visibility: hidden;
}}

@media(max-width: 900px) {{
    .block-container {{
        padding: 1rem;
    }}
}}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("best_model.pkl not found")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError("feature_names.pkl not found")

    loaded_model = joblib.load(MODEL_PATH)
    features = list(joblib.load(FEATURE_PATH))

    return loaded_model, features


@st.cache_data
def load_interval():

    if not INTERVAL_PATH.exists():
        return None

    try:
        radius = float(joblib.load(INTERVAL_PATH))

        if np.isfinite(radius) and radius >= 0:
            return radius

    except Exception:
        pass

    return None


try:
    model, feature_names = load_model()
    interval_radius = load_interval()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# =====================================================
# PREDICTION FUNCTION
# =====================================================

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

    if data["functioning_day"] == 0:
        return 0

    prediction = float(model.predict(input_df)[0])

    if not np.isfinite(prediction):
        raise ValueError("Invalid model prediction")

    return max(0, round(prediction))


# =====================================================
# ACTUAL VS PREDICTED EVALUATION
# =====================================================

@st.cache_data
def evaluate_model():

    if not DATA_PATH.exists():
        raise FileNotFoundError("dataset/train.csv not found")

    df = pd.read_csv(DATA_PATH)

    required = {
        "Date", "Hour", "Season", "Bikes_Rented"
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df = df.dropna(subset=["Date"])

    df = df.sort_values(
        by=["Date", "Hour"]
    ).reset_index(drop=True)

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    df["Season"] = df["Season"].fillna(
        df["Season"].mode()[0]
    )

    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    df["IsRushHour"] = df["Hour"].isin(
        [7, 8, 9, 17, 18, 19]
    ).astype(int)

    df = pd.get_dummies(
        df,
        columns=["Season"],
        dtype=int
    )

    df = df.drop(
        columns=["Id", "Date"],
        errors="ignore"
    )

    y = pd.to_numeric(
        df["Bikes_Rented"],
        errors="coerce"
    )

    X = df.drop(
        columns=["Bikes_Rented"]
    ).apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median()
    ).fillna(0)

    valid = y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if len(X) < 10:
        raise ValueError("Not enough valid dataset rows")

    calibration_end = int(len(X) * 0.85)

    X_test = X.iloc[calibration_end:]
    y_test = y.iloc[calibration_end:]

    X_test = X_test.reindex(
        columns=feature_names,
        fill_value=0
    )

    predictions = np.maximum(
        model.predict(X_test),
        0
    )

    results = pd.DataFrame({

        "Sample": range(1, len(y_test) + 1),
        "Actual": y_test.values,
        "Predicted": predictions

    })

    metrics = {

        "r2": r2_score(y_test, predictions),

        "mae": mean_absolute_error(
            y_test,
            predictions
        ),

        "rmse": np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

    }

    return metrics, results


try:
    metrics, test_results = evaluate_model()

except Exception as error:
    metrics = None
    test_results = None
    evaluation_error = str(error)


# =====================================================
# PLOT FUNCTIONS
# =====================================================

def chart_layout(fig, height=350):

    fig.update_layout(

        height=height,

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20
        ),

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color=TEXT,
            family="Inter"
        ),

        hovermode="x unified"

    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="rgba(109,165,192,.15)",
        zeroline=False
    )

    return fig


def actual_prediction_chart():

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test_results["Sample"],
            y=test_results["Actual"],
            name="Actual Demand",
            mode="lines",
            line=dict(
                color=BLUE,
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_results["Sample"],
            y=test_results["Predicted"],
            name="Predicted Demand",
            mode="lines",
            line=dict(
                color="#27C7BE",
                width=2
            )
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Bike Rentals",
        xaxis_title="Chronological Test Samples",
        yaxis_title="Number of Bike Rentals"
    )

    return chart_layout(fig, 390)


def demand_range_chart(minimum, expected, maximum):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=["Minimum", "Expected", "Maximum"],

            y=[minimum, expected, maximum],

            mode="lines+markers+text",

            line=dict(
                color=ACCENT,
                width=3
            ),

            marker=dict(
                size=13,
                color=[TEAL, BLUE, ACCENT]
            ),

            text=[
                f"{minimum:,}",
                f"{expected:,}",
                f"{maximum:,}"
            ],

            textposition="top center",

            fill="tozeroy",

            fillcolor="rgba(15,150,156,.12)",

            showlegend=False

        )
    )

    fig.update_layout(
        title="Predicted Demand Range",
        xaxis_title="Demand Category",
        yaxis_title="Bike Rentals"
    )

    return chart_layout(fig)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🚲</div>
            <div class="brand-name">
                Ride<span>Predict</span>
            </div>
            <div class="brand-caption">
                BIKE RENTAL DEMAND
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Prediction",
            "Analysis",
            "Model Insights",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### MODEL PERFORMANCE")

    if metrics is not None:

        st.metric(
            "R² Score",
            f"{metrics['r2'] * 100:.2f}%"
        )

        st.metric(
            "MAE",
            f"{metrics['mae']:.2f}"
        )

        st.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}"
        )

    else:

        st.warning(
            "Evaluation unavailable"
        )

    st.caption(
        f"Algorithm: {type(model).__name__}"
    )

    st.caption("RidePredict v3.0")


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="main-title">
        Bike Rental
        <span class="title-accent">
            Demand Prediction
        </span>
    </div>

    <div class="subtitle">
        Predict bike rental demand using weather,
        time and calendar information.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# PREDICTION PAGE
# =====================================================

if page == "Prediction":

    left, right = st.columns(
        [1.05, 0.95],
        gap="large"
    )

    with left:

        st.markdown(
            '<div class="panel-title">▤ Enter Prediction Details</div>',
            unsafe_allow_html=True
        )

        with st.form("prediction_form"):

            st.markdown(
                '<div class="section-label">◷ TIME & CALENDAR</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                hour = st.slider(
                    "Hour",
                    0,
                    23,
                    9
                )

            with c2:

                months = [
                    "January", "February", "March",
                    "April", "May", "June",
                    "July", "August", "September",
                    "October", "November", "December"
                ]

                month_name = st.selectbox(
                    "Month",
                    months,
                    index=5
                )

            with c3:

                day = st.selectbox(
                    "Day",
                    list(range(1, 32)),
                    index=14
                )

            with c4:

                weekdays = [
                    "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday",
                    "Saturday", "Sunday"
                ]

                weekday = st.selectbox(
                    "Day of Week",
                    weekdays
                )

            st.markdown(
                '<div class="section-label">☁ WEATHER CONDITIONS</div>',
                unsafe_allow_html=True
            )

            w1, w2, w3, w4 = st.columns(4)

            with w1:

                temperature = st.number_input(
                    "Temperature (°C)",
                    -30.0,
                    60.0,
                    20.0
                )

                dew_point = st.number_input(
                    "Dew Point (°C)",
                    -30.0,
                    40.0,
                    10.0
                )

            with w2:

                humidity = st.slider(
                    "Humidity (%)",
                    0,
                    100,
                    50
                )

                solar_radiation = st.number_input(
                    "Solar Radiation",
                    0.0,
                    1500.0,
                    0.0
                )

            with w3:

                wind_speed = st.number_input(
                    "Wind Speed (m/s)",
                    0.0,
                    100.0,
                    2.0
                )

                rainfall = st.number_input(
                    "Rainfall (mm)",
                    0.0,
                    500.0,
                    0.0
                )

            with w4:

                visibility = st.number_input(
                    "Visibility (m)",
                    0.0,
                    2000.0,
                    1000.0
                )

                snowfall = st.number_input(
                    "Snowfall (mm)",
                    0.0,
                    100.0,
                    0.0
                )

            st.markdown(
                '<div class="section-label">▱ OTHER INFORMATION</div>',
                unsafe_allow_html=True
            )

            o1, o2, o3 = st.columns(3)

            with o1:

                season = st.selectbox(
                    "Season",
                    [
                        "Spring",
                        "Summer",
                        "Autumn",
                        "Winter"
                    ]
                )

            with o2:

                holiday = st.selectbox(
                    "Holiday",
                    ["No", "Yes"]
                )

            with o3:

                functioning_day = st.selectbox(
                    "Functioning Day",
                    ["Yes", "No"]
                )

            submitted = st.form_submit_button(
                "🚲 Predict Bike Demand",
                use_container_width=True
            )

    with right:

        st.markdown(
            '<div class="panel-title">▥ Prediction Results</div>',
            unsafe_allow_html=True
        )

        if submitted:

            try:

                month = months.index(month_name) + 1
                day_of_week = weekdays.index(weekday)

                data = {

                    "hour": hour,

                    "temperature": temperature,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "visibility": visibility,
                    "dew_point": dew_point,
                    "solar_radiation": solar_radiation,
                    "rainfall": rainfall,
                    "snowfall": snowfall,

                    "holiday": int(holiday == "Yes"),

                    "functioning_day": int(
                        functioning_day == "Yes"
                    ),

                    "month": month,
                    "day": day,
                    "day_of_week": day_of_week,
                    "season": season

                }

                expected = predict_demand(data)

                if interval_radius is not None:

                    minimum = max(
                        0,
                        round(expected - interval_radius)
                    )

                    maximum = max(
                        expected,
                        round(expected + interval_radius)
                    )

                else:

                    minimum = expected
                    maximum = expected

                st.session_state["prediction"] = {

                    "minimum": minimum,
                    "expected": expected,
                    "maximum": maximum,
                    "calibrated": interval_radius is not None

                }

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

        result = st.session_state.get("prediction")

        if result:

            minimum = result["minimum"]
            expected = result["expected"]
            maximum = result["maximum"]

            k1, k2, k3 = st.columns(3)

            with k1:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">MINIMUM DEMAND</div>
                        <div class="metric-value">{minimum:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with k2:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">EXPECTED DEMAND</div>
                        <div class="metric-value expected">{expected:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with k3:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">MAXIMUM DEMAND</div>
                        <div class="metric-value maximum">{maximum:,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("#### Predicted Demand Range")

            st.plotly_chart(
                demand_range_chart(
                    minimum,
                    expected,
                    maximum
                ),
                use_container_width=True,
                config={"displayModeBar": False}
            )

            if result["calibrated"]:

                st.markdown(
                    """
                    <div class="note">
                    <b>Prediction Note</b><br>
                    The displayed range uses the saved calibration
                    error radius. It is an approximate interval,
                    not a guarantee.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.warning(
                    "Calibration file unavailable. "
                    "Only expected demand is available."
                )

        else:

            st.info(
                "Enter details and click Predict Bike Demand."
            )

    # ACTUAL VS PREDICTED GRAPH

    st.divider()

    st.markdown("### Actual vs Predicted Demand")

    st.caption(
        "Genuine predictions on the final chronological "
        "15% test partition of train.csv."
    )

    if test_results is not None:

        st.plotly_chart(
            actual_prediction_chart(),
            use_container_width=True
        )

    else:

        st.warning(
            f"Evaluation graph unavailable: {evaluation_error}"
        )


# =====================================================
# ANALYSIS PAGE
# =====================================================

elif page == "Analysis":

    st.markdown("## Model Analysis")

    if metrics is not None:

        a, b, c = st.columns(3)

        a.metric(
            "R² Score",
            f"{metrics['r2'] * 100:.2f}%"
        )

        b.metric(
            "MAE",
            f"{metrics['mae']:.2f}"
        )

        c.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}"
        )

    if test_results is not None:

        st.markdown("### Actual vs Predicted")

        st.plotly_chart(
            actual_prediction_chart(),
            use_container_width=True
        )

    if IMPORTANCE_PATH.exists():

        importance = pd.read_csv(
            IMPORTANCE_PATH
        )

        importance = importance.sort_values(
            "Importance",
            ascending=True
        ).tail(10)

        fig = go.Figure(
            go.Bar(
                x=importance["Importance"],
                y=importance["Feature"],
                orientation="h",
                marker_color=ACCENT
            )
        )

        fig.update_layout(
            title="Top 10 Feature Importance",
            xaxis_title="Importance",
            yaxis_title=""
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    else:

        st.warning(
            "models/feature_importance.csv not found."
        )


# =====================================================
# MODEL INSIGHTS
# =====================================================

elif page == "Model Insights":

    st.markdown("## Model Insights")

    if metrics is not None:

        a, b, c = st.columns(3)

        a.metric(
            "R² Score",
            f"{metrics['r2'] * 100:.2f}%"
        )

        b.metric(
            "MAE",
            f"{metrics['mae']:.2f}"
        )

        c.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}"
        )

    st.divider()

    st.markdown("### Model Configuration")

    st.write(
        f"Algorithm: {type(model).__name__}"
    )

    st.write(
        f"Features: {len(feature_names)}"
    )

    if interval_radius is not None:

        st.success(
            f"Calibration error radius: ±{interval_radius:.2f}"
        )

    st.markdown("### Features Used")

    st.code(
        ", ".join(feature_names)
    )


# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "About":

    st.markdown("## About RidePredict")

    st.write(
        """
        RidePredict is a machine-learning application
        designed to estimate bike rental demand.

        It uses:

        - Weather conditions
        - Time and calendar information
        - Seasonal variations
        - Holiday information
        - Operating-day information

        The application uses an Extra Trees Regressor
        trained on historical bike rental data.
        """
    )

    st.markdown("### Technology Stack")

    st.write(
        "Python • Streamlit • Scikit-learn • Pandas • "
        "NumPy • Plotly • Joblib"
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    f"""
    <div style="
        text-align:center;
        color:{MUTED};
        font-size:11px;
        padding:10px;
    ">
        RidePredict | Bike Rental Demand Prediction
        <br>
        Built with Python & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)