import os
from dotenv import load_dotenv
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "genai-agent")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load and split docs
DOCS_FOLDER = Path("docs/")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_documents = []

for file in DOCS_FOLDER.glob("*.txt"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.split_text(text)
    all_documents.extend([
        {"id": f"{file.stem}-chunk-{i}", "text": chunk, "source": file.name}
        for i, chunk in enumerate(chunks)
    ])

for file in DOCS_FOLDER.glob("*.md"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = splitter.split_text(text)
    all_documents.extend([
        {"id": f"{file.stem}-chunk-{i}", "text": chunk, "source": file.name}
        for i, chunk in enumerate(chunks)
    ])

if not all_documents:
    print("⚠️ No .txt or .md files found in /docs/.")
    exit()

texts = [doc["text"] for doc in all_documents]
metadatas = [{"source": doc["source"]} for doc in all_documents]
ids = [doc["id"] for doc in all_documents]

print(f"🔄 Generating embeddings for {len(texts)} chunks...")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# ✅ Use index_name, NOT index object
vectorstore = PineconeVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    index_name=PINECONE_INDEX_NAME,
    namespace=PINECONE_NAMESPACE,
    ids=ids
)

print(f"✅ Successfully upserted {len(texts)} chunks to index '{PINECONE_INDEX_NAME}' (namespace: '{PINECONE_NAMESPACE}')")
