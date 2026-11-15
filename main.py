
import os
import requests
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

from flask import Flask, request, render_template_string

app = Flask(__name__)

# Project files
DATA_FILE = "agricultural_monthly_app_data.csv"
MODEL_FILE = "agricultural_price_prediction_model.pkl"

MODEL_URL = (
    "https://github.com/Donaakshay/Agricultural-Price-Prediction/"
    "releases/download/v1.0/agricultural_price_prediction_model.pkl"
)

# Load dataset
data = pd.read_csv(DATA_FILE)
data["Month_Date"] = pd.to_datetime(data["Month_Date"])
data = data.sort_values("Month_Date")


# Download model from GitHub Release if not available locally
def load_model():
    if not os.path.exists(MODEL_FILE):
        response = requests.get(MODEL_URL, timeout=180)
        response.raise_for_status()

        with open(MODEL_FILE, "wb") as file:
            file.write(response.content)

    return joblib.load(MODEL_FILE)


artifact = load_model()


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Agricultural Price Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f7f2;
            color: #26352a;
        }

        header {
            background: #174d2b;
            color: white;
            padding: 25px;
            text-align: center;
        }

        .container {
            max-width: 1150px;
            margin: auto;
            padding: 20px;
        }

        .filters, .card, .chart {
            background: white;
            padding: 22px;
            margin-bottom: 22px;
            border-radius: 12px;
            box-shadow: 0 2px 10px #00000012;
        }

        .filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
        }

        select, button {
            width: 100%;
            padding: 11px;
            border-radius: 7px;
            border: 1px solid #ccc;
        }

        button {
            background: #237442;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            margin-top: 24px;
        }

        button:hover {
            background: #174d2b;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .metric-value {
            font-size: 25px;
            font-weight: bold;
            color: #237442;
        }

        h2 {
            color: #174d2b;
        }

        footer {
            text-align: center;
            padding: 25px;
            color: #666;
        }
    </style>
</head>

<body>

<header>
    <h1>🌾 Agricultural Market Price Prediction</h1>
    <p>Historical Market Analysis and Machine Learning Predictions</p>
</header>

<div class="container">

    <form method="GET" class="filters">

        <div>
            <label>Select Crop</label>
            <select name="crop">
                {% for x in crops %}
                <option value="{{x}}" {% if x == crop %}selected{% endif %}>
                    {{x}}
                </option>
                {% endfor %}
            </select>
        </div>

        <div>
            <label>Select Market</label>
            <select name="market">
                {% for x in markets %}
                <option value="{{x}}" {% if x == market %}selected{% endif %}>
                    {{x}}
                </option>
                {% endfor %}
            </select>
        </div>

        <div>
            <label>Select Yard</label>
            <select name="yard">
                {% for x in yards %}
                <option value="{{x}}" {% if x == yard %}selected{% endif %}>
                    {{x}}
                </option>
                {% endfor %}
            </select>
        </div>

        <div>
            <label>Select Variety</label>
            <select name="variety">
                {% for x in varieties %}
                <option value="{{x}}" {% if x == variety %}selected{% endif %}>
                    {{x}}
                </option>
                {% endfor %}
            </select>
        </div>

        <div>
            <button type="submit">Analyze Market</button>
        </div>

    </form>

    {% if latest is not none %}

    <h2>{{ crop }} - {{ market }}</h2>

    <div class="metrics">

        <div class="card">
            <h3>Latest Recorded Price (₹/Quintal)</h3>
            <div class="metric-value">{{ latest_price }}</div>
        </div>

        <div class="card">
            <h3>Latest Recorded Arrivals</h3>
            <div class="metric-value">{{ arrivals }}</div>
        </div>

        <div class="card">
            <h3>Latest Data Month</h3>
            <div class="metric-value">{{ latest_month }}</div>
        </div>

    </div>

    <div class="chart">
        <h2>Historical Monthly Price Trend</h2>
        {{ price_chart | safe }}
    </div>

    <div class="chart">
        <h2>Monthly Market Arrivals</h2>
        {{ arrivals_chart | safe }}
    </div>

    <div class="chart">
        <h2>Seasonal Price Analysis</h2>
        {{ seasonal_chart | safe }}
    </div>

    <h2>Next-Month Price Prediction</h2>

    <div class="metrics">

        <div class="card">
            <h3>Previous Recorded Price</h3>
            <div class="metric-value">{{ previous_price }}</div>
        </div>

        <div class="card">
            <h3>Predicted Price (Extra Trees)</h3>
            <div class="metric-value">{{ extra_price }}</div>
        </div>

        <div class="card">
            <h3>Predicted Price</h3>
            <div class="metric-value">{{ predicted_price }}</div>
        </div>

    </div>

    <p>
        Prediction month: <b>{{ forecast_month }}</b>
    </p>

    {% else %}
        <div class="card">
            <h2>No historical price records available.</h2>
            <p>Try another crop or market selection.</p>
        </div>
    {% endif %}

</div>

<footer>
    Agricultural Market Price Analysis | Machine Learning Project
    <br>
    Historical-data demonstration. Predictions are not live market quotes
    or guaranteed future prices.
</footer>

</body>
</html>
"""


@app.route("/")
def home():

    crops = sorted(data["CommName"].dropna().unique())

    crop = request.args.get("crop", crops[0])

    crop_data = data[data["CommName"] == crop]

    markets = sorted(crop_data["AmcName"].dropna().unique())

    market = request.args.get(
        "market",
        markets[0] if len(markets) else ""
    )

    market_data = crop_data[crop_data["AmcName"] == market]

    yards = sorted(market_data["YardName"].dropna().unique())

    yard = request.args.get(
        "yard",
        yards[0] if len(yards) else ""
    )

    yard_data = market_data[market_data["YardName"] == yard]

    varieties = sorted(
        yard_data["VarityName"].dropna().unique()
    )

    variety = request.args.get(
        "variety",
        varieties[0] if len(varieties) else ""
    )

    selected = yard_data[
        yard_data["VarityName"] == variety
    ].copy()

    selected = selected.sort_values("Month_Date")

    history = selected.dropna(
        subset=["Model_Median"]
    ).copy()

    if history.empty:

        return render_template_string(
            HTML,
            crops=crops,
            markets=markets,
            yards=yards,
            varieties=varieties,
            crop=crop,
            market=market,
            yard=yard,
            variety=variety,
            latest=None
        )

    latest = history.iloc[-1]
    latest_date = latest["Month_Date"]

    # Historical price chart
    fig_price = px.line(
        history,
        x="Month_Date",
        y="Model_Median",
        markers=True,
        labels={
            "Month_Date": "Month",
            "Model_Median": "Price (₹/Quintal)"
        }
    )

    # Arrivals chart
    fig_arrivals = px.bar(
        history,
        x="Month_Date",
        y="Arrivals_Total",
        labels={
            "Month_Date": "Month",
            "Arrivals_Total": "Total Arrivals"
        }
    )

    # Seasonal analysis
    seasonal = history.copy()
    seasonal["Calendar Month"] = (
        seasonal["Month_Date"].dt.month
    )

    seasonal_avg = (
        seasonal.groupby("Calendar Month")["Model_Median"]
        .mean()
        .reset_index()
    )

    seasonal_avg["Month"] = pd.to_datetime(
        seasonal_avg["Calendar Month"].astype(str),
        format="%m"
    ).dt.strftime("%B")

    fig_season = px.bar(
        seasonal_avg,
        x="Month",
        y="Model_Median",
        labels={
            "Month": "Calendar Month",
            "Model_Median": "Average Price (₹/Quintal)"
        }
    )

    # Next-month prediction
    forecast_date = (
        latest_date + pd.DateOffset(months=1)
    )

    price_history = history.set_index(
        "Month_Date"
    )["Model_Median"]

    def get_lag(months):
        date = forecast_date - pd.DateOffset(
            months=months
        )
        return price_history.get(date, np.nan)

    lag_1 = get_lag(1)
    lag_2 = get_lag(2)
    lag_3 = get_lag(3)
    lag_6 = get_lag(6)
    lag_12 = get_lag(12)

    rolling_3 = [
        get_lag(i) for i in [1, 2, 3]
    ]

    rolling_6 = [
        get_lag(i) for i in range(1, 7)
    ]

    rolling_mean_3 = (
        np.mean(rolling_3)
        if not np.isnan(rolling_3).any()
        else np.nan
    )

    rolling_mean_6 = (
        np.mean(rolling_6)
        if not np.isnan(rolling_6).any()
        else np.nan
    )

    month = forecast_date.month

    feature_values = {
        "Month": month,
        "Quarter": forecast_date.quarter,
        "Month_Sin": np.sin(2 * np.pi * month / 12),
        "Month_Cos": np.cos(2 * np.pi * month / 12),
        "Lag_1": lag_1,
        "Lag_2": lag_2,
        "Lag_3": lag_3,
        "Lag_6": lag_6,
        "Lag_12": lag_12,
        "Rolling_Mean_3": rolling_mean_3,
        "Rolling_Mean_6": rolling_mean_6,
        "CommCode": latest["CommCode"],
        "AmcCode": latest["AmcCode"],
        "YardCode": latest["YardCode"],
        "VarityCode": latest["VarityCode"]
    }

    input_df = pd.DataFrame(
        [[feature_values[f] for f in artifact["all_features"]]],
        columns=artifact["all_features"]
    )

    input_df[artifact["numeric_features"]] = (
        artifact["imputer"].transform(
            input_df[artifact["numeric_features"]]
        )
    )

    extra_prediction = artifact["model"].predict(
        input_df.to_numpy()
    )[0]

    weights = artifact["blend_weights"]

    predicted_price = (
        weights["previous_month_baseline"]
        * latest["Model_Median"]
        +
        weights["extra_trees"]
        * extra_prediction
    )

    return render_template_string(
        HTML,

        crops=crops,
        markets=markets,
        yards=yards,
        varieties=varieties,

        crop=crop,
        market=market,
        yard=yard,
        variety=variety,

        latest=latest,

        latest_price=f"₹{latest['Model_Median']:,.2f}",
        arrivals=f"{latest['Arrivals_Total']:,.2f}",
        latest_month=latest_date.strftime("%B %Y"),

        price_chart=fig_price.to_html(
            full_html=False,
            include_plotlyjs=False
        ),

        arrivals_chart=fig_arrivals.to_html(
            full_html=False,
            include_plotlyjs=False
        ),

        seasonal_chart=fig_season.to_html(
            full_html=False,
            include_plotlyjs=False
        ),

        previous_price=f"₹{latest['Model_Median']:,.2f}",
        extra_price=f"₹{extra_prediction:,.2f}",
        predicted_price=f"₹{predicted_price:,.2f}",

        forecast_month=forecast_date.strftime("%B %Y")
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
