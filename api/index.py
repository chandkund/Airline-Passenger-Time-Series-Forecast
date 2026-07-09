import base64
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.forecast import generate_forecast
from src.evaluate import evaluate_forecast

app = Flask(__name__)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Airline Passenger Time Series Forecast</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.6rem; }
  .subtitle { color: #555; margin-bottom: 1.5rem; }
  .controls { margin: 1.5rem 0; }
  .controls label { font-weight: 600; margin-right: 0.5rem; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { text-align: left; padding: 0.4rem 0.8rem; border-bottom: 1px solid #eee; }
  .metrics { display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap; }
  .metric { background: #f4f4f6; border-radius: 8px; padding: 0.8rem 1.2rem; }
  .metric-label { font-size: 0.8rem; color: #666; }
  .metric-value { font-size: 1.4rem; font-weight: 700; }
  details { margin-top: 1rem; }
</style>
</head>
<body>
  <h1>Airline Passenger Time Series Forecast</h1>
  <p class="subtitle">ARIMA-style (Holt-Winters) forecast on the AirPassengers dataset.</p>

  <form class="controls" method="get">
    <label for="periods">Forecast horizon (months)</label>
    <input type="range" id="periods" name="periods" min="6" max="36" step="1" value="{{ periods }}"
           oninput="document.getElementById('periods-value').textContent = this.value">
    <span id="periods-value">{{ periods }}</span>
    <button type="submit">Update</button>
  </form>

  <img src="data:image/png;base64,{{ chart_b64 }}" alt="Forecast chart">

  <h2>Evaluation Metrics</h2>
  <div class="metrics">{{ metric_cards | safe }}</div>

  <details>
    <summary>Forecast Table</summary>
    <table>
      <thead><tr><th>Month</th><th>Forecast</th></tr></thead>
      <tbody>{{ forecast_rows | safe }}</tbody>
    </table>
  </details>
</body>
</html>
"""


@app.route("/")
def index():
    periods = request.args.get("periods", default=12, type=int)
    periods = max(6, min(36, periods))

    raw_data = load_data()
    processed = preprocess_data(raw_data)
    model = train_model(processed)
    forecast = generate_forecast(model, processed, periods=periods)
    metrics = evaluate_forecast(processed, model, forecast)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(processed.index, processed["Passengers"], label="Actual", marker="o")
    ax.plot(forecast.index, forecast, label="Forecast", marker="o")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passengers")
    ax.set_title("Airline Passenger Forecast")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    chart_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    forecast_rows = "".join(
        f"<tr><td>{idx.strftime('%Y-%m')}</td><td>{value:.1f}</td></tr>"
        for idx, value in forecast.items()
    )
    metric_cards = "".join(
        f'<div class="metric"><div class="metric-label">{name}</div>'
        f'<div class="metric-value">{value:.2f}</div></div>'
        for name, value in metrics.items()
    )

    return render_template_string(
        PAGE_TEMPLATE,
        periods=periods,
        chart_b64=chart_b64,
        forecast_rows=forecast_rows,
        metric_cards=metric_cards,
    )
