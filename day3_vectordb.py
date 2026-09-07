import os
import numpy as np
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client= genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chroma_client=chromadb.Client()
collection=chroma_client.create_collection(name="legal_test")
documents = [
    "The tenant must pay rent.",
    "The lessee owes a monthly payment.",
    "The sky is blue today."
]
ids=["clause_1", "clause_2", "clause_3"]
print("Embedding documents and saving to ChromaDB...")
embeddings = []
for doc in documents:
    res = client.models.embed_content(model="gemini-embedding-2", contents=doc)
    embeddings.append(res.embeddings[0].values)

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)
print("Success! Data stored in vector space.") 

query_text = "How much does the apartment cost?"
print(f"\nSearching database for: '{query_text}'")

query_res = client.models.embed_content(model="gemini-embedding-2", contents=query_text)
query_embedding = query_res.embeddings[0].values

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\n--- Search Results ---")
for i, doc in enumerate(results['documents'][0]):
    distance = results['distances'][0][i]
    print(f"Match {i+1}: {doc} (Distance: {distance:.4f})")