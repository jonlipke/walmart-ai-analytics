import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import plotly.graph_objects as go
from dotenv import load_dotenv
from pathlib import Path

# --- Load API key ---
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

# --- Setup paths ---
BASE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Data")

# --- Load data ---
df = pd.read_csv(BASE_DIR / "Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- Filter to Store 1 ---
store_df = df[df['Store'] == 1].copy().sort_values('Date').reset_index(drop=True)

# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
store_df['week_of_year'] = store_df['Date'].dt.isocalendar().week.astype(int)
store_df['month'] = store_df['Date'].dt.month
store_df['year'] = store_df['Date'].dt.year

# Lag features
store_df['sales_lag_1'] = store_df['Weekly_Sales'].shift(1)
store_df['sales_lag_4'] = store_df['Weekly_Sales'].shift(4)
store_df['sales_lag_52'] = store_df['Weekly_Sales'].shift(52)

# Drop rows with NaN from lag features
store_df = store_df.dropna().reset_index(drop=True)

print(f"Rows available after lag features: {len(store_df)}")

# -------------------------------------------------------------------
# TRAIN / TEST SPLIT
# Last 12 weeks = test set, everything before = training
# -------------------------------------------------------------------
TEST_WEEKS = 12
train = store_df.iloc[:-TEST_WEEKS]
test = store_df.iloc[-TEST_WEEKS:]

print(f"Training rows: {len(train)}")
print(f"Test rows: {len(test)}")

# --- Define features ---
FEATURES = [
    'week_of_year', 'month', 'year',
    'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'sales_lag_1', 'sales_lag_4', 'sales_lag_52'
]
TARGET = 'Weekly_Sales'

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

# -------------------------------------------------------------------
# TRAIN XGBOOST MODEL
# -------------------------------------------------------------------
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

# --- Predictions ---
xgb_predictions = xgb_model.predict(X_test)

# --- XGBoost metrics ---
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
print(f"\nXGBoost Results:")
print(f"  RMSE: ${xgb_rmse:,.0f}")
print(f"  MAE:  ${xgb_mae:,.0f}")

# -------------------------------------------------------------------
# TRAIN PROPHET MODEL ON SAME DATA FOR COMPARISON
# -------------------------------------------------------------------
print("\nTraining Prophet model for comparison...")
prophet_df = store_df[['Date', 'Weekly_Sales']].rename(
    columns={'Date': 'ds', 'Weekly_Sales': 'y'}
)
prophet_train = prophet_df.iloc[:-TEST_WEEKS]
prophet_test = prophet_df.iloc[-TEST_WEEKS:]

prophet_model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    changepoint_prior_scale=0.1
)
prophet_model.fit(prophet_train)

prophet_forecast = prophet_model.predict(prophet_test[['ds']])
prophet_predictions = prophet_forecast['yhat'].values

# --- Prophet metrics ---
prophet_rmse = np.sqrt(mean_squared_error(y_test, prophet_predictions))
prophet_mae = mean_absolute_error(y_test, prophet_predictions)
print(f"\nProphet Results:")
print(f"  RMSE: ${prophet_rmse:,.0f}")
print(f"  MAE:  ${prophet_mae:,.0f}")

# -------------------------------------------------------------------
# COMPARISON SUMMARY
# -------------------------------------------------------------------
print("\n" + "="*50)
print("MODEL COMPARISON — Store 1 (last 12 weeks)")
print("="*50)
print(f"{'Metric':<10} {'XGBoost':>15} {'Prophet':>15} {'Winner':>10}")
print("-"*50)
print(f"{'RMSE':<10} ${xgb_rmse:>13,.0f} ${prophet_rmse:>13,.0f} {'XGBoost' if xgb_rmse < prophet_rmse else 'Prophet':>10}")
print(f"{'MAE':<10} ${xgb_mae:>13,.0f} ${prophet_mae:>13,.0f} {'XGBoost' if xgb_mae < prophet_mae else 'Prophet':>10}")

# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------
print("\nFeature Importance (XGBoost):")
importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:<20} {bar} {row['Importance']:.4f}")

# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------
fig = go.Figure()

# Actuals
fig.add_scatter(
    x=test['Date'], y=y_test,
    name='Actuals', line=dict(color='steelblue', width=2)
)

# XGBoost predictions
fig.add_scatter(
    x=test['Date'], y=xgb_predictions,
    name=f'XGBoost (RMSE: ${xgb_rmse:,.0f})',
    line=dict(color='darkorange', width=2)
)

# Prophet predictions
fig.add_scatter(
    x=test['Date'], y=prophet_predictions,
    name=f'Prophet (RMSE: ${prophet_rmse:,.0f})',
    line=dict(color='green', width=2, dash='dash')
)

fig.update_layout(
    title='Walmart Store 1 — XGBoost vs Prophet (Last 12 Weeks)',
    xaxis_title='Week',
    yaxis_title='Weekly Sales ($)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)
fig.show()

# --- Save comparison results ---
output_path = BASE_DIR / "store1_model_comparison.csv"
comparison_df = pd.DataFrame({
    'Week': test['Date'].values,
    'Actual_Sales': y_test.values,
    'XGBoost_Forecast': xgb_predictions,
    'Prophet_Forecast': prophet_predictions
})
comparison_df.to_csv(output_path, index=False)
print(f"\nComparison results saved to: {output_path}")