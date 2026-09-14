import os
import streamlit as st
import pymupdf
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Page Configuration & Styling
st.set_page_config(page_title="Legal AI Agent", page_icon="⚖️", layout="wide")

# Custom CSS for a cleaner look
# Custom CSS compatible with both Dark and Light modes
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatMessage { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)
# 2. Resource Caching
@st.cache_resource
def init_system():
    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="ui_agent_db")
    except:
        pass
    collection = chroma_client.create_collection(name="ui_agent_db")
    
    return client, collection

client, collection = init_system()

# 3. Define the Agent Tool with UI Intercept
def search_contract(search_query: str) -> str:
    """Searches the uploaded contract for specific clauses, liabilities, or rules."""
    # Capture the agent's thought process for the UI
    if "current_searches" not in st.session_state:
        st.session_state.current_searches = []
    st.session_state.current_searches.append(search_query)

    query_res = client.models.embed_content(model="gemini-embedding-2", contents=search_query)
    results = collection.query(
        query_embeddings=[query_res.embeddings[0].values],
        n_results=1
    )
    if results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0][0]
    return "No relevant clauses found in the document."

# 4. Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            tools=[search_contract],
            temperature=0.2,
            system_instruction="You are a legal AI agent. Search the contract to warn the user of hidden liabilities. Max 2 searches per query. Format your output nicely with Markdown."
        )
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_chunks" not in st.session_state:
    st.session_state.extracted_chunks = []

# 5. Sidebar: Upload & Visualizations
with st.sidebar:
    st.header("📂 Document Pipeline")
    uploaded_file = st.file_uploader("Upload Legal Contract (PDF)", type=["pdf"])
    
    if uploaded_file and not st.session_state.get("pdf_processed"):
        with st.status("Processing Document...", expanded=True) as status:
            st.write("Reading PDF file...")
            
            # 1. Read directly from memory (Skip the temp_path writing/deleting entirely!)
            file_bytes = uploaded_file.read()
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            
            st.write("Chunking legal clauses...")
            chunks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                for b in page.get_text("blocks"):
                    if b[6] == 0:
                        text = " ".join(b[4].split()).strip()
                        if len(text) > 40:
                            chunks.append(text)
            
            st.session_state.extracted_chunks = chunks
            
            st.write("Generating vector embeddings...")
            embeddings = []
            ids = []
            for i, chunk in enumerate(chunks):
                res = client.models.embed_content(model="gemini-embedding-2", contents=chunk)
                embeddings.append(res.embeddings[0].values)
                ids.append(f"clause_{i}")
            
            st.write("Ingesting into ChromaDB...")
            collection.add(documents=chunks, embeddings=embeddings, ids=ids)
            st.session_state.pdf_processed = True
            
            # 2. Close the document properly to free up RAM
            doc.close()
            
            status.update(label="Document vectorized successfully!", state="complete", expanded=False)

    # Sidebar Metrics Dashboard
    if st.session_state.get("pdf_processed"):
        st.divider()
        st.subheader("📊 Database Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Clauses Found", len(st.session_state.extracted_chunks))
        col2.metric("Vector Dimensions", "768") # Standard Gemini embedding size
        
        with st.expander("👁️ View Raw Database Chunks"):
            for i, chunk in enumerate(st.session_state.extracted_chunks):
                st.markdown(f"**Chunk {i+1}**")
                st.caption(chunk)
                st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# 6. Main Chat Interface
st.title("⚖️ Legal Liability Extractor")
st.markdown("Upload a contract in the sidebar, then ask the AI agent to uncover hidden risks, penalties, and obligations.")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If the agent made tool calls during this past message, show them
        if msg["role"] == "assistant" and "tool_calls" in msg:
            with st.expander("🔍 View Agent Search Queries"):
                for search in msg["tool_calls"]:
                    st.code(f"Query: {search}", language="text")

# Chat Input
if prompt := st.chat_input("E.g., What are the financial penalties if I leak information?"):
    if not st.session_state.get("pdf_processed"):
        st.error("⚠️ Please upload a PDF contract in the sidebar first.")
    else:
        # User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Agent Response
        from google.genai.errors import ClientError # Add this import at the top of your file if you prefer
        
        with st.chat_message("assistant"):
            st.session_state.current_searches = [] # Reset search tracker
            
            with st.spinner("Agent is analyzing the vector database..."):
                try:
                    # We wrap the API call in a try block
                    response = st.session_state.agent.send_message(prompt)
                    st.markdown(response.text)
                    
                    # If the agent used tools, display them visually
                    if st.session_state.current_searches:
                        with st.expander("🔍 View Agent Search Queries", expanded=False):
                            for search in st.session_state.current_searches:
                                st.code(f"Query: {search}", language="text")
                                
                    # Save response and tool calls to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.text,
                        "tool_calls": st.session_state.current_searches
                    })
                    
                except ClientError as e:
                    # If we hit the rate limit, fail gracefully without crashing the UI
                    error_msg = str(e)
                    if "429" in error_msg:
                        st.warning("⚠️ **Rate Limit Exceeded:** The free-tier AI is resting. Please wait about 60 seconds and try your question again.")
                    else:
                        st.error(f"⚠️ **API Error:** {error_msg}")
        