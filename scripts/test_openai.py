from openai import OpenAI
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

try:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input="test"
    )
    print("✅ Success! Preview:", res.data[0].embedding[:5])
except Exception as e:
    print("❌ FAILED:", e)
