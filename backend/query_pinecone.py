import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from query_logic import get_contextual_answer

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Dummy embedding to satisfy LangChain constructor
embedding_fn = lambda text: [0.0] * 1536

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_fn,
    text_key="text",
    namespace=os.getenv("PINECONE_NAMESPACE", "genai-agent")
)

def get_top_reference(query: str) -> dict:
    try:
        query_vector = get_contextual_answer(query, vectorstore)
        return {
            "response": query_vector
        }
    except Exception as e:
        return {
            "error": str(e)
        }
