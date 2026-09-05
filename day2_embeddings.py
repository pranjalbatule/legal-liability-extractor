import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
sentence_1 = "The tenant must pay rent."
sentence_2 = "The lessee owes a monthly payment."
sentence_3 = "The sky is blue today."
print("Generating embeddings (converting text to numbers)...")

# Fetch embeddings using the new gemini-embedding-2 model
res_1 = client.models.embed_content(model="gemini-embedding-2", contents=sentence_1)
res_2 = client.models.embed_content(model="gemini-embedding-2", contents=sentence_2)
res_3 = client.models.embed_content(model="gemini-embedding-2", contents=sentence_3)

vec_1 = res_1.embeddings[0].values
vec_2 = res_2.embeddings[0].values
vec_3 = res_3.embeddings[0].values

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_and_2 = cosine_similarity(vec_1, vec_2)
sim_1_and_3 = cosine_similarity(vec_1, vec_3)

print("\n--- Cosine Similarity Results ---")
print(f"Match Score ('Rent' vs 'Payment'): {sim_1_and_2:.4f}")
print(f"Match Score ('Rent' vs 'Sky'):     {sim_1_and_3:.4f}")