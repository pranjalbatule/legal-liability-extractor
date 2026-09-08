import os
import pymupdf

def extract_and_chunk_pdf(file_path):
    print(f"Opening {file_path}...")
    doc = pymupdf.open(file_path)
    cleaned_chunks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        blocks = page.get_text("blocks")
        
        for b in blocks:
            
            if b[6] == 0:
                
                text = " ".join(b[4].split()).strip()
                if len(text) > 40:
                    cleaned_chunks.append(text)
                    
    return cleaned_chunks

file_name = "sample_contract.pdf"

if os.path.exists(file_name):
    chunks = extract_and_chunk_pdf(file_name)
    
    print(f"\nCreated {len(chunks)} complete clause chunks.\n")
    for idx, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---")
        print(f"{chunk}\n")
else:
    print(f"Error: Could not find '{file_name}'.")