from query_logic import get_contextual_answer
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone client and index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Dummy embedding (actual embedding is handled in query_logic)
embedding_fn = lambda text: [0.0] * 1536

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_fn,
    text_key="text",
    namespace=os.getenv("PINECONE_NAMESPACE", "genai-agent")
)

def handle_n8n_input(payload: dict) -> dict:
    try:
        query = payload.get("query", "")
        if not query:
            return {"error": "No query provided."}

        answer = get_contextual_answer(query, vectorstore)
        return {
            "query": query,
            "answer": answer
        }
    except Exception as e:
        return {"error": str(e)}
