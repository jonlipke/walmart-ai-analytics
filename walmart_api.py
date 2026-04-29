import anthropic
import os
import json
import pandas as pd
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import uvicorn

# -------------------------------------------------------------------
# LOAD ENVIRONMENT
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# DATA PATHS — works both locally and on Render
# -------------------------------------------------------------------
# On Render, data files live in the same directory as the script
# Locally, they live in the Data folder
BASE_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent / "data"))

INSIGHTS_FILE = BASE_DIR / "all_stores_insights.csv"
FORECAST_FILE = BASE_DIR / "all_stores_forecast.csv"
ANALYSES_DIR  = BASE_DIR / "analyses"

# -------------------------------------------------------------------
# LOAD DATA AT STARTUP
# -------------------------------------------------------------------
insights_df = pd.read_csv(INSIGHTS_FILE)
forecast_df = pd.read_csv(FORECAST_FILE)

print(f"Data loaded successfully")
print(f"  Insights:  {len(insights_df)} stores")
print(f"  Forecasts: {len(forecast_df)} rows")

# -------------------------------------------------------------------
# TOOL IMPLEMENTATIONS (embedded directly — no subprocess needed)
# -------------------------------------------------------------------
def get_portfolio_summary() -> str:
    """Returns a high level summary of all 45 Walmart stores."""
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


def get_store_profile(store_num: int) -> str:
    """Returns detailed profile for a specific Walmart store."""
    store = insights_df[insights_df['store'] == store_num]
    if store.empty:
        return f"Store {store_num} not found"

    narrative = ""
    analysis_file = ANALYSES_DIR / f"store{store_num}_analysis.txt"
    if analysis_file.exists():
        with open(analysis_file, 'r', encoding='utf-8') as f:
            narrative = f.read()

    profile = store.iloc[0].to_dict()
    profile['narrative_analysis'] = narrative
    profile['store'] = int(profile['store'])
    return json.dumps(profile, indent=2)


def get_store_forecast(store_num: int, weeks: int = 12) -> str:
    """Returns forecast data for a specific store."""
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


def get_declining_stores() -> str:
    """Returns all stores with a declining sales trend."""
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


def get_high_risk_stores() -> str:
    """Returns all stores classified as high risk."""
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


# -------------------------------------------------------------------
# TOOL REGISTRY — maps tool names to functions and schemas
# -------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_portfolio_summary",
        "description": """Returns a high level summary of all 45 Walmart stores including
        performance tier distribution, risk levels, trend directions,
        model comparison results, and primary sales drivers. Use this when
        the user asks about the overall portfolio, all stores, or wants a summary.""",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_store_profile",
        "description": """Returns detailed profile for a specific Walmart store including
        performance tier, risk level, trend, key insight, recommended action,
        forecast outlook, and model accuracy metrics. Use this when the user
        asks about a specific store number.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_num": {"type": "integer", "description": "Store number between 1 and 45"}
            },
            "required": ["store_num"]
        }
    },
    {
        "name": "get_store_forecast",
        "description": """Returns forecast data for a specific store including actual sales
        history and predicted future sales with confidence intervals. Use this
        when the user wants to see forecast numbers or sales trends for a
        specific store.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "store_num": {"type": "integer", "description": "Store number between 1 and 45"},
                "weeks": {"type": "integer", "description": "Number of weeks to return (default 12)", "default": 12}
            },
            "required": ["store_num"]
        }
    },
    {
        "name": "get_declining_stores",
        "description": """Returns all stores with a declining sales trend including their
        average weekly sales, risk level, and key insight. Use this when
        the user asks which stores are declining, underperforming, or at risk.""",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_high_risk_stores",
        "description": """Returns all stores classified as high risk with their performance
        tier, trend, and recommended action. Use this when the user asks about
        high risk stores or wants to know where to focus attention.""",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }
]

TOOL_FUNCTIONS = {
    "get_portfolio_summary": lambda args: get_portfolio_summary(),
    "get_store_profile": lambda args: get_store_profile(**args),
    "get_store_forecast": lambda args: get_store_forecast(**args),
    "get_declining_stores": lambda args: get_declining_stores(),
    "get_high_risk_stores": lambda args: get_high_risk_stores(),
}

# -------------------------------------------------------------------
# CORE QUERY FUNCTION
# -------------------------------------------------------------------
def run_query(question: str) -> tuple[str, list[str]]:
    """Run a question through Claude with embedded tools."""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages = [{"role": "user", "content": question}]
    tools_called = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_args = tool_use.input
                tools_called.append(f"{tool_name} {json.dumps(tool_args)}")

                # Call the embedded tool function
                tool_fn = TOOL_FUNCTIONS.get(tool_name)
                if tool_fn:
                    result_text = tool_fn(tool_args)
                else:
                    result_text = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_text
                })

            messages.append({
                "role": "user",
                "content": tool_results
            })

        elif response.stop_reason == "end_turn":
            answer = next(
                (b.text for b in response.content if hasattr(b, 'text')),
                "No response"
            )
            return answer, tools_called

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="Walmart Analytics AI API",
    description="Conversational AI interface for Walmart store analytics powered by Claude",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    tools_called: list[str]

# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "name": "Walmart Analytics AI API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Ask any question about the Walmart portfolio",
            "GET /portfolio": "Get portfolio summary",
            "GET /stores/high-risk": "Get high risk stores",
            "GET /stores/declining": "Get declining stores",
            "GET /stores/{store_num}": "Get profile for a specific store",
            "GET /stores/{store_num}/forecast": "Get forecast for a specific store",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        answer, tools_called = run_query(request.question)
        return QueryResponse(
            question=request.question,
            answer=answer,
            tools_called=tools_called
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio")
def get_portfolio():
    try:
        answer, tools_called = run_query("Give me a concise portfolio summary of all Walmart stores")
        return {"answer": answer, "tools_called": tools_called}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/high-risk")
def get_high_risk():
    try:
        answer, tools_called = run_query("Which stores are high risk and what should we do about them?")
        return {"answer": answer, "tools_called": tools_called}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/declining")
def get_declining():
    try:
        answer, tools_called = run_query("Which stores are declining and what do they have in common?")
        return {"answer": answer, "tools_called": tools_called}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/{store_num}")
def get_store(store_num: int):
    if store_num < 1 or store_num > 45:
        raise HTTPException(status_code=400, detail="Store number must be between 1 and 45")
    try:
        answer, tools_called = run_query(
            f"Tell me everything about Store {store_num} including its risk level, trend, and key insights"
        )
        return {"store": store_num, "answer": answer, "tools_called": tools_called}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/{store_num}/forecast")
def get_store_forecast_endpoint(store_num: int):
    if store_num < 1 or store_num > 45:
        raise HTTPException(status_code=400, detail="Store number must be between 1 and 45")
    try:
        answer, tools_called = run_query(
            f"Show me the sales forecast for Store {store_num} and explain what it means"
        )
        return {"store": store_num, "answer": answer, "tools_called": tools_called}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  WALMART ANALYTICS AI API")
    print("="*60)
    print("Starting server at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))