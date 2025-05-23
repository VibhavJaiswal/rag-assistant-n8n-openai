import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ✅ Ensure .env is loaded from the current file's directory
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(dotenv_path)

# ✅ Print for debugging (optional but helps trace issues)
print("🔑 OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
print("🏢 ORG_ID =", os.getenv("OPENAI_ORG_ID"))
print("📁 PROJECT_ID =", os.getenv("OPENAI_PROJECT_ID"))

# ✅ Initialize client with proper environment context
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

# 🔤 Replace with your actual texts
texts_to_embed = [
    "Vector databases are optimized for storing vector representations of data.",
    "OpenAI provides powerful models like GPT and embedding APIs.",
    "Pinecone is a vector database used in RAG systems for fast similarity search.",
]

def embed_batch(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in tqdm(texts, desc="Generating embeddings"):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            vectors.append(response.data[0].embedding)
        except Exception as e:
            print(f"❌ Failed to embed: {text[:30]}... ➜ {e}")
            vectors.append([])
    return vectors


if __name__ == "__main__":
    embeddings = embed_batch(texts_to_embed)

    # 🔍 Show preview of first few vectors
    for i, vec in enumerate(embeddings[:3]):
        print(f"\nText {i+1}: {texts_to_embed[i]}\nEmbedding Preview: {vec[:5]}")
