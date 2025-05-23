import requests

url = "http://127.0.0.1:8000/rag-search"
query = {"query": "What is RAG?"}

try:
    response = requests.post(url, json=query)
    response.raise_for_status()
    data = response.json()
    print("✅ Success:", data.get("response", "-"))
except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
except ValueError:
    print("❌ Response was not JSON.")
