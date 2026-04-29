import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------------------------
# User settings
# -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Data")

INPUT_FILE = BASE_DIR / "sales_data.csv"
OUTPUT_FILE = BASE_DIR / "sales_forecast.csv"

# --- Load and filter data ---
df = pd.read_csv(INPUT_FILE)
df = df[df['Store'] == 1]                        # filter to Store 1
df['ds'] = pd.to_datetime(df['ds'])
df = df[['ds', 'y']].sort_values('ds')           # keep only what Prophet needs

# --- Train model ---
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    changepoint_prior_scale=0.1
)
model.fit(df)

# --- Generate forecast ---
future = model.make_future_dataframe(periods=12, freq='W')  # 12 weeks out, weekly grain
forecast = model.predict(future)

# --- Quick sanity check plot ---
fig = go.Figure()
fig.add_scatter(x=df['ds'], y=df['y'], name='Actuals', line=dict(color='steelblue'))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='darkorange'))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                fill=None, line=dict(color='rgba(0,0,0,0)'), showlegend=False)
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                fill='tonexty', fillcolor='rgba(255,165,0,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='Confidence interval')
fig.show()

# --- Export to CSV for Tableau ---
output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
output = output.rename(columns={
    'ds': 'Week',
    'yhat': 'Forecast_Sales',
    'yhat_lower': 'Forecast_Lower',
    'yhat_upper': 'Forecast_Upper'
})

# Join actuals back in where they exist
actuals = df.rename(columns={'ds': 'Week', 'y': 'Actual_Sales'})
output = output.merge(actuals, on='Week', how='left')

# Round to 2 decimal places for cleanliness
output = output.round(2)

output.to_csv(OUTPUT_FILE, index=False)
print(f"Forecast exported to: {OUTPUT_FILE}")