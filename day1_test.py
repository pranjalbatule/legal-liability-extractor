import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

legal_clause = """
The Receiving Party shall indemnify, defend, and hold harmless the Disclosing Party 
from and against any and all losses, damages, liabilities, costs, and expenses 
arising out of any breach of this Confidentiality Agreement.
"""
prompt = f"Explain this legal clause in plain English, highlighting any severe risks to the Receiving Party:\n\n{legal_clause}"
response = client.models.generate_content(
    model="gemini-3.5-flash", 
    contents=prompt
)
print("--- AI Translation ---")
print(response.text)