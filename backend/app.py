import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from query_logic import get_contextual_answer

# Load environment
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = FastAPI()

# Enable CORS for all origins (you can restrict this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated environment values
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")  # ✅ updated
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
namespace = os.getenv("PINECONE_NAMESPACE")        # ✅ updated

# Load Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Set up LangChain vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index, embeddings, namespace=namespace)

@app.post("/rag-search")
async def rag_search(request: Request):
    body = await request.json()
    query = body.get("query", "")

    if not query:
        return {"error": "Missing query parameter."}

    # ✅ Add system context to the query
    enhanced_query = f"You are a helpful AI assistant. Please answer clearly. The user asked: {query}"

    # ✅ Use enhanced query for retrieval
    result = get_contextual_answer(enhanced_query, vectorstore)

    # ✅ Add references to response for frontend display
    references = []
    for i, r in enumerate(result.get("results", [])):
        source = r.get("metadata", {}).get("source", "Unknown source")
        references.append(f"({i + 1}) {source}")
    result["references"] = "\n".join(references)

    return result
