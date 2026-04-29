import anthropic
import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# --- Load API key ---
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

# --- Load data ---
BASE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Data")
df = pd.read_csv(BASE_DIR / "Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# --- Focus on Store 1 ---
store_df = df[df['Store'] == 1].sort_values('Date')

# --- Build summary to send Claude ---
summary = f"""
Walmart Store 1 — Weekly Sales Data Summary

Date range: {store_df['Date'].min().date()} to {store_df['Date'].max().date()}
Total weeks: {len(store_df)}
Average weekly sales: ${store_df['Weekly_Sales'].mean():,.0f}
Best week: ${store_df['Weekly_Sales'].max():,.0f} on {store_df.loc[store_df['Weekly_Sales'].idxmax(), 'Date'].date()}
Worst week: ${store_df['Weekly_Sales'].min():,.0f} on {store_df.loc[store_df['Weekly_Sales'].idxmin(), 'Date'].date()}
Holiday weeks: {store_df['Holiday_Flag'].sum()} of {len(store_df)} total weeks
Average unemployment: {store_df['Unemployment'].mean():.2f}%
Unemployment range: {store_df['Unemployment'].min():.2f}% to {store_df['Unemployment'].max():.2f}%
Average fuel price: ${store_df['Fuel_Price'].mean():.3f}
Average temperature: {store_df['Temperature'].mean():.1f}F

Full weekly data:
{store_df.to_string(index=False)}
"""

# --- Call Claude ---
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""You are a retail analytics expert. Analyze this Walmart store data 
        and provide:
        1. The three most significant patterns you observe
        2. Which external factors (temperature, fuel price, CPI, unemployment) 
           appear most correlated with sales performance
        3. Two specific, actionable recommendations for store management
        4. Any anomalies or weeks that stand out and possible explanations
        
        Be specific — reference actual dates, numbers, and percentages from the data.
        
        {summary}"""
    }]
)

print(response.content[0].text)

# --- Save output ---
output_path = BASE_DIR / "store1_analysis.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(response.content[0].text)
print(f"\nAnalysis saved to: {output_path}")