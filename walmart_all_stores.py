import anthropic
import os
import pandas as pd
from prophet import Prophet
from dotenv import load_dotenv
from pathlib import Path

# --- Load API key ---
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

# --- Setup paths ---
BASE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Data")
FORECAST_DIR = BASE_DIR / "forecasts"
ANALYSIS_DIR = BASE_DIR / "analyses"

# Create output folders if they don't exist
FORECAST_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

# --- Load full dataset ---
df = pd.read_csv(BASE_DIR / "Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- Initialize Anthropic client ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- Store results for combined output ---
all_forecasts = []

# --- Loop through all 45 stores ---
for store_num in range(1, 46):
    print(f"\nProcessing Store {store_num} of 45...")

    # Filter to this store
    store_df = df[df['Store'] == store_num].copy()
    store_df = store_df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    store_df = store_df[['ds', 'y']].sort_values('ds')

    # --- Train Prophet model ---
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.1
    )
    model.fit(store_df)

    # --- Generate forecast ---
    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)

    # --- Build output dataframe ---
    output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    output = output.rename(columns={
        'ds': 'Week',
        'yhat': 'Forecast_Sales',
        'yhat_lower': 'Forecast_Lower',
        'yhat_upper': 'Forecast_Upper'
    })

    # Join actuals back in
    actuals = store_df.rename(columns={'ds': 'Week', 'y': 'Actual_Sales'})
    output = output.merge(actuals, on='Week', how='left')
    output['Store'] = store_num
    output = output.round(2)

    # Save individual store forecast
    output.to_csv(FORECAST_DIR / f"store{store_num}_forecast.csv", index=False)

    # Add to combined output
    all_forecasts.append(output)

    # --- Claude analysis ---
    last_actual = store_df['y'].iloc[-1]
    first_forecast = forecast['yhat'].iloc[len(store_df)]
    pct_change = ((first_forecast - last_actual) / last_actual) * 100

    summary = f"""
    Walmart Store {store_num} Summary
    Date range: {store_df['ds'].min().date()} to {store_df['ds'].max().date()}
    Average weekly sales: ${store_df['y'].mean():,.0f}
    Best week: ${store_df['y'].max():,.0f}
    Worst week: ${store_df['y'].min():,.0f}
    Last actual week: ${last_actual:,.0f}
    First forecasted week: ${first_forecast:,.0f}
    Projected change: {pct_change:.1f}%
    Average unemployment: {df[df['Store']==store_num]['Unemployment'].mean():.2f}%
    Average fuel price: ${df[df['Store']==store_num]['Fuel_Price'].mean():.3f}
    Average CPI: {df[df['Store']==store_num]['CPI'].mean():.2f}
    Average temperature: {df[df['Store']==store_num]['Temperature'].mean():.1f}F
    """

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""You are a retail analytics expert. Analyze this Walmart store 
            and provide a concise 3-4 sentence summary covering:
            1. Overall sales trend and performance level
            2. Most notable external factor influence
            3. Forecast outlook and any risk factors
            
            Be specific with numbers. Store {store_num} data:
            {summary}"""
        }]
    )

    # Save individual store analysis
    analysis_path = ANALYSIS_DIR / f"store{store_num}_analysis.txt"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"WALMART STORE {store_num} ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(response.content[0].text)