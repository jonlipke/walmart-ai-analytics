import anthropic
import os
import json
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Load API key ---
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

SERVER_SCRIPT = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\walmart_mcp_server.py")
PYTHON_PATH = r"C:\Program Files\Python314\python.exe"

async def run_query(question: str):
    """Send a question to Claude with access to Walmart MCP tools."""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    server_params = StdioServerParameters(
        command=PYTHON_PATH,
        args=[str(SERVER_SCRIPT)]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get available tools
            tools_result = await session.list_tools()

            # Convert MCP tools to Anthropic format
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools_result.tools
            ]

            print(f"\nQuestion: {question}")
            print("-" * 60)

            messages = [{"role": "user", "content": question}]

            # Agentic loop
            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    tools=tools,
                    messages=messages
                )

                # Claude wants to use a tool
                if response.stop_reason == "tool_use":
                    tool_uses = [b for b in response.content if b.type == "tool_use"]

                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    tool_results = []
                    for tool_use in tool_uses:
                        print(f"  → Claude calling: {tool_use.name} {tool_use.input}")

                        result = await session.call_tool(
                            tool_use.name,
                            tool_use.input
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result.content[0].text
                        })

                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                # Claude is done
                elif response.stop_reason == "end_turn":
                    final = next(
                        (b.text for b in response.content if hasattr(b, 'text')),
                        "No response"
                    )
                    print(f"\nClaude: {final}")
                    break

# -------------------------------------------------------------------
# TEST QUESTIONS
# -------------------------------------------------------------------
async def main():
    questions = [
        "Give me a summary of the Walmart store portfolio",
        "Which stores are high risk and what should we do about them?",
        "Tell me about Store 7 — why is it declining?",
        "Which stores are declining and what do they have in common?",
        "Show me the forecast for Store 28"
    ]

    for question in questions:
        await run_query(question)
        print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())