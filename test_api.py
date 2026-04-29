import anthropic
import os
from dotenv import load_dotenv
from pathlib import Path

# Point directly to your .env file
load_dotenv(Path(r"C:\Users\jonli\OneDrive\Documents\Python Scripts\Claude Projects\Code\.env"))

# Verify it loaded
key = os.environ.get("ANTHROPIC_API_KEY")
print(f"Key found: {key is not None}")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=64,
    messages=[{
        "role": "user",
        "content": "Say 'API connection successful' and nothing else."
    }]
)

print(response.content[0].text)