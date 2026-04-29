import anthropic
import os
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Load API key ---
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

SERVER_SCRIPT = Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\walmart_mcp_server.py")
PYTHON_PATH = r"C:\Program Files\Python314\python.exe"

async def run_query(question: str, session: ClientSession, tools: list) -> str:
    """Send a single question to Claude with access to Walmart MCP tools."""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            tools=tools,
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
                print(f"  → Calling: {tool_use.name} {tool_use.input}")

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

        elif response.stop_reason == "end_turn":
            return next(
                (b.text for b in response.content if hasattr(b, 'text')),
                "No response"
            )

async def main():
    """Interactive chat loop with Walmart MCP server."""

    server_params = StdioServerParameters(
        command=PYTHON_PATH,
        args=[str(SERVER_SCRIPT)]
    )

    print("\n" + "="*60)
    print("  WALMART ANALYTICS AI ASSISTANT")
    print("="*60)
    print("Connecting to Walmart MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get available tools
            tools_result = await session.list_tools()
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools_result.tools
            ]

            print(f"Connected — {len(tools)} tools available")
            print("\nAvailable tools:")
            for tool in tools:
                print(f"  • {tool['name']}")

            print("\nYou can ask questions like:")
            print("  • Give me a portfolio summary")
            print("  • Which stores are high risk?")
            print("  • Tell me about Store 7")
            print("  • Which stores are declining?")
            print("  • Show me the forecast for Store 28")
            print("\nType 'quit' or 'exit' to stop")
            print("="*60)

            # Interactive loop
            while True:
                print()
                try:
                    question = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                print()
                response = await run_query(question, session, tools)
                print(f"\nClaude: {response}")
                print("\n" + "-"*60)

if __name__ == "__main__":
    asyncio.run(main())