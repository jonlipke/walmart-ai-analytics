import json
import os
import tempfile
import pandas as pd
import pantab
import tableauserverclient as TSC
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# -------------------------------------------------------------------
# DATA PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Data")

INSIGHTS_FILE = BASE_DIR / "all_stores_insights.csv"
FORECAST_FILE = BASE_DIR / "all_stores_forecast.csv"
WALMART_FILE  = BASE_DIR / "Walmart_Sales.csv"
ANALYSES_DIR  = BASE_DIR / "analyses"

# -------------------------------------------------------------------
# LOAD DATA AT STARTUP
# -------------------------------------------------------------------
insights_df = pd.read_csv(INSIGHTS_FILE)
forecast_df = pd.read_csv(FORECAST_FILE)
walmart_df  = pd.read_csv(WALMART_FILE)
walmart_df['Date'] = pd.to_datetime(walmart_df['Date'])

print("Data loaded successfully")
print(f"  Insights:  {len(insights_df)} stores")
print(f"  Forecasts: {len(forecast_df)} rows")
print(f"  Raw data:  {len(walmart_df)} rows")

# Load Tableau credentials at startup
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

TABLEAU_SERVER     = os.environ.get("TABLEAU_SERVER")
TABLEAU_SITE       = os.environ.get("TABLEAU_SITE")
TABLEAU_PAT_NAME   = os.environ.get("TABLEAU_PAT_NAME")
TABLEAU_PAT_SECRET = os.environ.get("TABLEAU_PAT_SECRET")

# -------------------------------------------------------------------
# INITIALIZE MCP SERVER
# -------------------------------------------------------------------
mcp = FastMCP("walmart-analytics")

# -------------------------------------------------------------------
# DEFINE TOOLS
# -------------------------------------------------------------------
@mcp.tool()
def get_portfolio_summary() -> str:
    """Returns a high level summary of all 45 Walmart stores including 
    performance tier distribution, risk levels, trend directions, 
    model comparison results, and primary sales drivers. Use this when 
    the user asks about the overall portfolio, all stores, or wants a summary."""
    
    summary = {
        "total_stores": len(insights_df),
        "performance_tiers": insights_df['performance_tier'].value_counts().to_dict(),
        "risk_levels": insights_df['risk_level'].value_counts().to_dict(),
        "overall_trends": insights_df['overall_trend'].value_counts().to_dict(),
        "better_model_counts": insights_df['better_model'].value_counts().to_dict(),
        "primary_drivers": insights_df['primary_sales_driver'].value_counts().to_dict(),
        "avg_weekly_sales_portfolio": round(insights_df['avg_weekly_sales'].mean(), 2),
        "highest_sales_store": int(insights_df.loc[insights_df['avg_weekly_sales'].idxmax(), 'store']),
        "lowest_sales_store": int(insights_df.loc[insights_df['avg_weekly_sales'].idxmin(), 'store']),
        "high_risk_stores": insights_df[insights_df['risk_level'] == 'high']['store'].tolist(),
        "declining_stores": insights_df[insights_df['overall_trend'] == 'declining']['store'].tolist(),
        "improving_stores": insights_df[insights_df['overall_trend'] == 'improving']['store'].tolist()
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_store_profile(store_num: int) -> str:
    """Returns detailed profile for a specific Walmart store including 
    performance tier, risk level, trend, key insight, recommended action, 
    forecast outlook, and model accuracy metrics. Use this when the user 
    asks about a specific store number."""
    
    store = insights_df[insights_df['store'] == store_num]

    if store.empty:
        return f"Store {store_num} not found"

    # Load narrative analysis if available
    narrative = ""
    analysis_file = ANALYSES_DIR / f"store{store_num}_analysis.txt"
    if analysis_file.exists():
        with open(analysis_file, 'r', encoding='utf-8') as f:
            narrative = f.read()

    profile = store.iloc[0].to_dict()
    profile['narrative_analysis'] = narrative
    profile['store'] = int(profile['store'])

    return json.dumps(profile, indent=2)


@mcp.tool()
def get_store_forecast(store_num: int, weeks: int = 12) -> str:
    """Returns forecast data for a specific store including actual sales 
    history and predicted future sales with confidence intervals. Use this 
    when the user wants to see forecast numbers or sales trends for a 
    specific store."""
    
    store_forecast = forecast_df[forecast_df['Store'] == store_num].copy()

    if store_forecast.empty:
        return f"No forecast data for Store {store_num}"

    recent = store_forecast.tail(weeks)

    result = {
        "store": store_num,
        "weeks_returned": len(recent),
        "data": recent.to_dict(orient='records')
    }

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_declining_stores() -> str:
    """Returns all stores with a declining sales trend including their 
    average weekly sales, risk level, and key insight. Use this when 
    the user asks which stores are declining, underperforming, or at risk."""
    
    declining = insights_df[insights_df['overall_trend'] == 'declining'][[
        'store', 'performance_tier', 'risk_level',
        'avg_weekly_sales', 'key_insight', 'recommended_action'
    ]].copy()
    declining['store'] = declining['store'].astype(int)

    result = {
        "total_declining": len(declining),
        "stores": declining.to_dict(orient='records')
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_high_risk_stores() -> str:
    """Returns all stores classified as high risk with their performance 
    tier, trend, and recommended action. Use this when the user asks about 
    high risk stores or wants to know where to focus attention."""
    
    high_risk = insights_df[insights_df['risk_level'] == 'high'][[
        'store', 'performance_tier', 'overall_trend',
        'avg_weekly_sales', 'key_insight', 'recommended_action'
    ]].copy()
    high_risk['store'] = high_risk['store'].astype(int)

    result = {
        "total_high_risk": len(high_risk),
        "stores": high_risk.to_dict(orient='records')
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def refresh_tableau_datasource() -> str:
    """Refreshes the Walmart Store Insights data source on Tableau Cloud
    with the latest forecast and analysis data. Use this when the user
    asks to update Tableau, refresh the dashboard, or push new data to
    Tableau Cloud."""

    try:
        # Load latest insights data
        df = pd.read_csv(INSIGHTS_FILE)
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]

        # Create Hyper extract in temp location
        temp_hyper = Path(tempfile.mkdtemp()) / "walmart_insights.hyper"
        pantab.frame_to_hyper(df, temp_hyper, table="Extract")

        # Authenticate to Tableau Cloud
        tableau_auth = TSC.PersonalAccessTokenAuth(
            TABLEAU_PAT_NAME,
            TABLEAU_PAT_SECRET,
            site_id=TABLEAU_SITE
        )
        server = TSC.Server(TABLEAU_SERVER, use_server_version=True)

        with server.auth.sign_in(tableau_auth):
            # Find the existing data source
            all_datasources, _ = server.datasources.get()
            datasource = next(
                (ds for ds in all_datasources if ds.name == "Walmart Store Insights"),
                None
            )

            if not datasource:
                return json.dumps({
                    "status": "error",
                    "message": "Walmart Store Insights data source not found on Tableau Cloud"
                })

            # Find project
            all_projects, _ = server.projects.get()
            project = next(
                (p for p in all_projects if p.id == datasource.project_id),
                all_projects[0]
            )

            # Republish with latest data
            ds_item = TSC.DatasourceItem(project.id, name="Walmart Store Insights")
            published = server.datasources.publish(
                ds_item,
                str(temp_hyper),
                TSC.Server.PublishMode.Overwrite
            )

            result = {
                "status": "success",
                "message": "Walmart Store Insights refreshed on Tableau Cloud",
                "datasource_id": published.id,
                "rows_published": len(df),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Walmart MCP Server...")
    mcp.run(transport="stdio")
