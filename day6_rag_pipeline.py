import os
import pymupdf
import chromadb
from google import genai
from dotenv import load_dotenv

# 1. Setup API and Database
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Initializing database and ingesting contract...")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="final_rag")

# Quickly extract and embed the PDF chunks (From Days 4 & 5)
doc = pymupdf.open("sample_contract.pdf")
chunks = []
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    for b in page.get_text("blocks"):
        if b[6] == 0:
            text = " ".join(b[4].split()).strip()
            if len(text) > 40:
                chunks.append(text)

embeddings = []
ids = []
for i, chunk in enumerate(chunks):
    res = client.models.embed_content(model="gemini-embedding-2", contents=chunk)
    embeddings.append(res.embeddings[0].values)
    ids.append(f"clause_{i}")

collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# 2. Retrieval: Finding the relevant clause
query_text = "What are the financial penalties if I leak information?"
print(f"\nUser Question: '{query_text}'")

query_res = client.models.embed_content(model="gemini-embedding-2", contents=query_text)
results = collection.query(
    query_embeddings=[query_res.embeddings[0].values],
    n_results=1
)
retrieved_clause = results['documents'][0][0]

print("\n--- Retrieved Raw Legalese ---")
print(retrieved_clause)

# 3. Generation: Translating to Plain English
print("\n--- AI Plain English Translation ---")

# We construct a strict prompt forcing the AI to only use the retrieved text
prompt = f"""
You are an expert legal AI assistant.
A user asked this question: "{query_text}"

Based ONLY on the following legal clause extracted from their contract, answer the user's question in simple, plain English. 
Clearly warn them about any hidden financial liabilities. Keep it concise.

LEGAL CLAUSE:
{retrieved_clause}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print(response.text)