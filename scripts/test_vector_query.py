from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Constants from your .env
index_name = os.environ.get("PINECONE_INDEX_NAME")
namespace = os.environ.get("PINECONE_NAMESPACE")

# Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vectorstore client
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace,
)

# Run test query
query = "what is vector database"
results = vectorstore.similarity_search_with_score(query, k=3)

# Print results
for i, (doc, score) in enumerate(results):
    print(f"[{i}] Score: {score}")
    print(f"Text: {doc.page_content}\n")
