// ==========================================
// RIDEPREDICT - COMPLETE JAVASCRIPT
// ==========================================


// HTML ELEMENTS

const predictionForm = document.getElementById("predictionForm");

const predictButton = document.getElementById("predictButton");

const buttonText = document.getElementById("buttonText");

const predictionValue = document.getElementById("predictionValue");

const resultMessage = document.getElementById("resultMessage");

const predictionStatus = document.getElementById("predictionStatus");


// ==========================================
// RANGE SLIDERS
// ==========================================

const rangeFields = [
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point",
    "solar_radiation",
    "rainfall",
    "snowfall"
];


function updateRangeValue(slider) {

    const display = document.getElementById(slider.id + "Value");

    if (display) {
        display.textContent = slider.value;
    }

    // Update slider fill percentage

    const min = Number(slider.min);
    const max = Number(slider.max);
    const value = Number(slider.value);

    const percentage = ((value - min) / (max - min)) * 100;

    slider.style.background =
        `linear-gradient(90deg,
        #4775ff ${percentage}%,
        rgba(255,255,255,0.15) ${percentage}%)`;
}


rangeFields.forEach(function (field) {

    const slider = document.getElementById(field);

    updateRangeValue(slider);

    slider.addEventListener("input", function () {

        updateRangeValue(slider);

    });

});


// ==========================================
// GET FORM DATA
// ==========================================

function getFormData() {

    return {

        hour: Number(document.getElementById("hour").value),

        temperature: Number(document.getElementById("temperature").value),

        humidity: Number(document.getElementById("humidity").value),

        wind_speed: Number(document.getElementById("wind_speed").value),

        visibility: Number(document.getElementById("visibility").value),

        dew_point: Number(document.getElementById("dew_point").value),

        solar_radiation: Number(document.getElementById("solar_radiation").value),

        rainfall: Number(document.getElementById("rainfall").value),

        snowfall: Number(document.getElementById("snowfall").value),

        holiday: Number(document.getElementById("holiday").value),

        functioning_day: Number(document.getElementById("functioning_day").value),

        month: Number(document.getElementById("month").value),

        day: Number(document.getElementById("day").value),

        day_of_week: Number(document.getElementById("day_of_week").value),

        season: document.getElementById("season").value

    };

}


// ==========================================
// VALIDATION
// ==========================================

function validateInputs(data) {

    if (data.hour < 0 || data.hour > 23) {
        return "Hour must be between 0 and 23.";
    }

    if (data.month < 1 || data.month > 12) {
        return "Month must be between 1 and 12.";
    }

    if (data.day < 1 || data.day > 31) {
        return "Day must be between 1 and 31.";
    }

    if (data.humidity < 0 || data.humidity > 100) {
        return "Humidity must be between 0 and 100.";
    }

    const numericFields = [
        "temperature",
        "humidity",
        "wind_speed",
        "visibility",
        "dew_point",
        "solar_radiation",
        "rainfall",
        "snowfall"
    ];

    for (const field of numericFields) {

        if (!Number.isFinite(data[field])) {

            return `Invalid value for ${field}.`;

        }

    }

    if (
        data.wind_speed < 0 ||
        data.visibility < 0 ||
        data.solar_radiation < 0 ||
        data.rainfall < 0 ||
        data.snowfall < 0
    ) {

        return "Weather values cannot be negative.";

    }

    return null;

}


// ==========================================
// PREDICTION API
// ==========================================

predictionForm.addEventListener("submit", async function (event) {

    event.preventDefault();

    const data = getFormData();

    const validationError = validateInputs(data);

    if (validationError) {

        resultMessage.textContent = validationError;

        predictionStatus.textContent = "Invalid Input";

        return;

    }


    // Loading state

    predictButton.disabled = true;

    buttonText.textContent = "Generating Prediction...";

    predictionValue.textContent = "...";

    resultMessage.textContent =
        "Analyzing conditions using machine learning.";

    predictionStatus.textContent = "Processing";


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });


        const result = await response.json();


        if (!response.ok || !result.success) {

            throw new Error(
                result.error || "Prediction failed."
            );

        }


        const demand = Number(result.predicted_demand);


        if (!Number.isFinite(demand)) {

            throw new Error("Invalid prediction received.");

        }


        // DISPLAY RESULT

        predictionValue.textContent =
            demand.toLocaleString("en-IN");

        predictionStatus.textContent = "Completed";


        let demandLevel;


        if (demand < 100) {

            demandLevel = "Low";

        }

        else if (demand < 500) {

            demandLevel = "Moderate";

        }

        else {

            demandLevel = "High";

        }


        resultMessage.textContent =
            `The model predicts approximately ${demand.toLocaleString("en-IN")} bike rentals. Demand level: ${demandLevel}.`;


        // SAVE HISTORY

        savePrediction(data, demand);

        displayHistory();


    }

    catch (error) {

        console.error("Prediction Error:", error);

        predictionValue.textContent = "--";

        predictionStatus.textContent = "Failed";

        resultMessage.textContent = error.message;

    }

    finally {

        predictButton.disabled = false;

        buttonText.textContent = "Generate Prediction";

    }

});


// ==========================================
// SAVE PREDICTION HISTORY
// ==========================================

function savePrediction(data, demand) {

    try {

        let history = JSON.parse(

            localStorage.getItem("ridePredictHistory") || "[]"

        );


        history.unshift({

            demand: demand,

            hour: data.hour,

            temperature: data.temperature,

            date: new Date().toLocaleString()

        });


        // Keep only latest 10 predictions

        history = history.slice(0, 10);


        localStorage.setItem(

            "ridePredictHistory",

            JSON.stringify(history)

        );

    }

    catch (error) {

        console.error("History Storage Error:", error);

    }

}


// ==========================================
// DISPLAY HISTORY
// ==========================================

function displayHistory() {

    const historyList = document.getElementById("historyList");

    let history = [];


    try {

        history = JSON.parse(

            localStorage.getItem("ridePredictHistory") || "[]"

        );

    }

    catch (error) {

        console.error(error);

    }


    if (history.length === 0) {

        historyList.innerHTML =
            '<p class="empty-history">No predictions yet.</p>';

        return;

    }


    historyList.innerHTML = "";


    history.forEach(function (item) {

        const row = document.createElement("div");

        row.className = "history-item";


        const date = document.createElement("span");

        date.textContent = item.date;


        const details = document.createElement("span");

        details.textContent =
            `Hour: ${item.hour} | Temperature: ${item.temperature}°C`;


        const demand = document.createElement("strong");

        demand.textContent =
            `${Number(item.demand).toLocaleString("en-IN")} Bikes`;


        row.appendChild(date);

        row.appendChild(details);

        row.appendChild(demand);


        historyList.appendChild(row);

    });

}


// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener("DOMContentLoaded", function () {

    displayHistory();

    console.log("RidePredict loaded successfully.");

});