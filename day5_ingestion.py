import os
import pymupdf
import chromadb
from google import genai
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection(name="contract_clauses")
except:
    pass
collection = chroma_client.create_collection(name="contract_clauses")


def extract_chunks(file_path):
    doc = pymupdf.open(file_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] == 0:
                text = " ".join(b[4].split()).strip()
                if len(text) > 40:
                    chunks.append(text)
    return chunks

file_name = "sample_contract.pdf"
print(f"Extracting clauses from {file_name}...")
documents = extract_chunks(file_name)


print(f"Embedding {len(documents)} clauses and storing in ChromaDB...")
embeddings = []
ids = []
for i, doc in enumerate(documents):
    res = client.models.embed_content(model="gemini-embedding-2", contents=doc)
    embeddings.append(res.embeddings[0].values)
    ids.append(f"clause_{i}")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)
print("Success! Contract fully ingested into vector space.")


query_text = "What are the financial penalties if I leak information?"
print(f"\nSearching contract for: '{query_text}'")

query_res = client.models.embed_content(model="gemini-embedding-2", contents=query_text)
query_embedding = query_res.embeddings[0].values


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

print("\n--- Top Contract Match ---")
print(results['documents'][0][0])