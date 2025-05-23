import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

if __name__ == "__main__":
    text = "What is a vector database and how is it used?"
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        print(f"\nText: {text}\nEmbedding Preview: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
