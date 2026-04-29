import anthropic
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from dotenv import load_dotenv
from pathlib import Path

# --- Load API key ---
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

# --- Setup paths ---
BASE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Data")

# --- Load data ---
df = pd.read_csv(BASE_DIR / "Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- Initialize Anthropic client ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# --- Store results ---
all_store_insights = []

def analyze_store(store_num):
    """Run Prophet, XGBoost, and Claude analysis for a single store."""
    
    print(f"\nAnalyzing Store {store_num}...")
    
    # --- Filter to store ---
    store_df = df[df['Store'] == store_num].copy().sort_values('Date').reset_index(drop=True)

    # -------------------------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------------------------
    store_df['week_of_year'] = store_df['Date'].dt.isocalendar().week.astype(int)
    store_df['month'] = store_df['Date'].dt.month
    store_df['year'] = store_df['Date'].dt.year
    store_df['sales_lag_1'] = store_df['Weekly_Sales'].shift(1)
    store_df['sales_lag_4'] = store_df['Weekly_Sales'].shift(4)
    store_df['sales_lag_52'] = store_df['Weekly_Sales'].shift(52)
    store_df = store_df.dropna().reset_index(drop=True)

    TEST_WEEKS = 12
    train = store_df.iloc[:-TEST_WEEKS]
    test = store_df.iloc[-TEST_WEEKS:]

    FEATURES = [
        'week_of_year', 'month', 'year',
        'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'sales_lag_1', 'sales_lag_4', 'sales_lag_52'
    ]
    TARGET = 'Weekly_Sales'

    # -------------------------------------------------------------------
    # XGBOOST
    # -------------------------------------------------------------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(train[FEATURES], train[TARGET])
    xgb_predictions = xgb_model.predict(test[FEATURES])
    xgb_rmse = np.sqrt(mean_squared_error(test[TARGET], xgb_predictions))
    xgb_mae = mean_absolute_error(test[TARGET], xgb_predictions)

    # Feature importance
    importance = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    top_features = importance.head(3)['Feature'].tolist()

    # -------------------------------------------------------------------
    # PROPHET
    # -------------------------------------------------------------------
    prophet_df = store_df[['Date', 'Weekly_Sales']].rename(
        columns={'Date': 'ds', 'Weekly_Sales': 'y'}
    )
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        changepoint_prior_scale=0.1
    )
    prophet_model.fit(prophet_df.iloc[:-TEST_WEEKS])
    prophet_forecast = prophet_model.predict(
        prophet_df.iloc[-TEST_WEEKS:][['ds']]
    )
    prophet_rmse = np.sqrt(mean_squared_error(
        test[TARGET], prophet_forecast['yhat'].values
    ))
    prophet_mae = mean_absolute_error(
        test[TARGET], prophet_forecast['yhat'].values
    )

    # -------------------------------------------------------------------
    # CLAUDE STRUCTURED JSON ANALYSIS
    # -------------------------------------------------------------------
    avg_sales = store_df['Weekly_Sales'].mean()
    best_week = store_df['Weekly_Sales'].max()
    worst_week = store_df['Weekly_Sales'].min()
    last_actual = store_df['Weekly_Sales'].iloc[-1]
    avg_unemployment = store_df['Unemployment'].mean()
    avg_cpi = store_df['CPI'].mean()
    avg_fuel = store_df['Fuel_Price'].mean()
    pct_vs_avg = ((last_actual - avg_sales) / avg_sales) * 100

    prompt = f"""
    Analyze this Walmart store data and return ONLY a valid JSON object.
    No preamble, no explanation, no markdown code blocks. Just the raw JSON.

    Required JSON structure:
    {{
        "store": {store_num},
        "performance_tier": "high|mid|low",
        "overall_trend": "improving|stable|declining",
        "risk_level": "high|medium|low",
        "primary_sales_driver": "one of: seasonality|unemployment|CPI|fuel_price|temperature|holidays",
        "key_insight": "one specific sentence with actual numbers",
        "recommended_action": "one specific actionable recommendation",
        "anomaly_detected": true or false,
        "anomaly_description": "brief description or null if none",
        "forecast_outlook": "optimistic|neutral|cautious"
    }}

    Store {store_num} data:
    - Average weekly sales: ${avg_sales:,.0f}
    - Best week: ${best_week:,.0f}
    - Worst week: ${worst_week:,.0f}
    - Last actual week: ${last_actual:,.0f} ({pct_vs_avg:+.1f}% vs average)
    - Average unemployment: {avg_unemployment:.2f}%
    - Average CPI: {avg_cpi:.2f}
    - Average fuel price: ${avg_fuel:.3f}
    - Top XGBoost features: {', '.join(top_features)}
    - XGBoost RMSE: ${xgb_rmse:,.0f}
    - Prophet RMSE: ${prophet_rmse:,.0f}
    - Better model: {'XGBoost' if xgb_rmse < prophet_rmse else 'Prophet'}
    """

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    # --- Parse JSON response ---
    try:
        raw = response.content[0].text.strip()
        # Strip markdown code blocks if Claude adds them
        raw = raw.replace('```json', '').replace('```', '').strip()
        result = json.loads(raw)

        # Add model metrics to the result
        result['xgb_rmse'] = round(xgb_rmse, 2)
        result['xgb_mae'] = round(xgb_mae, 2)
        result['prophet_rmse'] = round(prophet_rmse, 2)
        result['prophet_mae'] = round(prophet_mae, 2)
        result['better_model'] = 'XGBoost' if xgb_rmse < prophet_rmse else 'Prophet'
        result['avg_weekly_sales'] = round(avg_sales, 2)
        result['top_xgb_features'] = ', '.join(top_features)

        print(f"  Store {store_num}: {result['performance_tier']} performer | "
              f"Risk: {result['risk_level']} | "
              f"Trend: {result['overall_trend']} | "
              f"Better model: {result['better_model']}")

        return result

    except json.JSONDecodeError as e:
        print(f"  Store {store_num}: JSON parse error — {e}")
        print(f"  Raw response: {raw[:200]}")
        return None

# -------------------------------------------------------------------
# RUN FOR ALL 45 STORES
# -------------------------------------------------------------------
print("Starting analysis for all 45 stores...")
print("This will take approximately 15-20 minutes\n")

for store_num in range(1, 46):
    result = analyze_store(store_num)
    if result:
        all_store_insights.append(result)

# -------------------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------------------

# Save as JSON
json_path = BASE_DIR / "all_stores_insights.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_store_insights, f, indent=2)
print(f"\nJSON saved to: {json_path}")

# Save as CSV for Tableau
insights_df = pd.DataFrame(all_store_insights)
csv_path = BASE_DIR / "all_stores_insights.csv"
insights_df.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")

# -------------------------------------------------------------------
# PORTFOLIO SUMMARY
# -------------------------------------------------------------------
print("\n" + "="*60)
print("WALMART PORTFOLIO SUMMARY")
print("="*60)

print(f"\nPerformance Tiers:")
print(insights_df['performance_tier'].value_counts().to_string())

print(f"\nRisk Levels:")
print(insights_df['risk_level'].value_counts().to_string())

print(f"\nOverall Trends:")
print(insights_df['overall_trend'].value_counts().to_string())

print(f"\nBetter Model (XGBoost vs Prophet):")
print(insights_df['better_model'].value_counts().to_string())

print(f"\nTop Primary Sales Drivers:")
print(insights_df['primary_sales_driver'].value_counts().to_string())

print(f"\nHigh Risk Stores:")
high_risk = insights_df[insights_df['risk_level'] == 'high'][['store', 'performance_tier', 'overall_trend', 'recommended_action']]
print(high_risk.to_string(index=False))

print(f"\nDeclining Stores:")
declining = insights_df[insights_df['overall_trend'] == 'declining'][['store', 'risk_level', 'avg_weekly_sales', 'key_insight']]
print(declining.to_string(index=False))