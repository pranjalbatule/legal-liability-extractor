import os
import pymupdf
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Initializing Agent memory and ingesting contract...")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="agent_db")


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


def search_contract(search_query: str) -> str:
    """
    Searches the user's uploaded legal contract for specific clauses, terms, or liabilities.
    Call this tool whenever you need to look up facts, rules, or penalties from the document.
    """
    print(f"\n[AGENT THOUGHT PROCESS] -> Decided to search database for: '{search_query}'")
    
    query_res = client.models.embed_content(model="gemini-embedding-2", contents=search_query)
    results = collection.query(
        query_embeddings=[query_res.embeddings[0].values],
        n_results=1
    )
    
    return results['documents'][0][0]

agent_chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        tools=[search_contract],
        temperature=0.2,
        system_instruction="""
        You are an elite legal AI agent. 
        CRITICAL INSTRUCTION: To conserve system resources, you must minimize your tool usage. 
        Do not make multiple single-word searches. Instead, combine your legal concepts into a single, broad search query. 
        Limit yourself to a maximum of TWO searches per user question. 
        Use the retrieved context to warn the user of hidden liabilities in plain English.
        """
    )
)




user_prompt = "What happens to my bank account if I accidentally leak information to a competitor?"
print(f"\nUser: {user_prompt}")


response = agent_chat.send_message(user_prompt)

print("\n--- Agent Final Answer ---")
print(response.text)

print("\n" + "="*50 + "\n")


follow_up_prompt = "Does that penalty still apply if the leak was a complete accident?"
print(f"User Follow-Up: {follow_up_prompt}")


follow_up_response = agent_chat.send_message(follow_up_prompt)

print("\n--- Agent Follow-Up Answer ---")
print(follow_up_response.text)