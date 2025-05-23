import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

# Load .env reliably
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

def embed_text(text: str) -> list[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")
    proj_id = os.getenv("OPENAI_PROJECT_ID")

    if not api_key or not org_id or not proj_id:
        raise EnvironmentError("❌ Missing one or more OpenAI env variables.")

    client = OpenAI(api_key=api_key, organization=org_id, project=proj_id)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_answer(user_query: str, references: list[tuple[Document, float]]) -> str:
    if not references:
        return "I don’t know based on the reference."

    reference_block = "\n\n".join(
        [f"[Reference {i+1}]: {doc.page_content}" for i, (doc, _) in enumerate(references)]
    )

    prompt = f"""You are a concise and accurate assistant. Use only the provided references to answer the user's question.

{reference_block}

[User Question]:
{user_query}

[Instructions]:
- Be brief but informative.
- Do not make up facts.
- If unsure, say "I don’t know based on the reference."

Answer:"""

    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")
    proj_id = os.getenv("OPENAI_PROJECT_ID")

    client = OpenAI(api_key=api_key, organization=org_id, project=proj_id)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who strictly follows reference-based context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

def get_contextual_answer(query: str, vectorstore: PineconeVectorStore) -> dict:
    try:
        query_vector = embed_text(query)
        namespace = os.getenv("NAMESPACE", "genai-agent")  # use env or fallback
        top_docs = vectorstore.similarity_search_by_vector_with_score(
            query_vector,
            k=3,
            namespace=namespace
        )

        print(f"🧪 Retrieved {len(top_docs)} docs from Pinecone.")

        if not top_docs:
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "results": []
            }

        answer = generate_answer(query, top_docs)

        results = [
            {
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            }
            for doc, score in top_docs
        ]

        return {
            "query": query,
            "answer": answer,
            "results": results
        }

    except Exception as e:
        return {
            "query": query,
            "answer": f"❌ Error during retrieval or generation: {str(e)}",
            "results": []
        }
