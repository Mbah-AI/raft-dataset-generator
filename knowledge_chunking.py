import pandas as pd
import uuid
import re
import os
import tqdm
from pypdf import PdfReader
from google.cloud import translate_v2

from constants import GCP_CRED_PATH, KNOWLEDGE_DATA_PATH, CHUNK_DATA_PATH

# Set the environment variable for the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CRED_PATH

# Initialize the Google Cloud Translate client
translator_client = translate_v2.Client()

# Load the knowledge data
knowledge_df = pd.read_csv(KNOWLEDGE_DATA_PATH)

# Initialize the chunk data
chunk_data = {
    "id": [],
    "content": [],
    "page": [],
    "knowledge_id": [],
    "title": [],
    "author": [],
    "organization": [],
    "published_year": []
}

# Set the character limit for each chunk
char_limit = 500

# Translate the knowledge data and chunk it
for index, row in tqdm.tqdm(knowledge_df.iterrows(), total=len(knowledge_df)):
    # Read the PDF file
    pdf_path = row["local_path"]
    reader = PdfReader(pdf_path)
    start_page = row["start_page"] - 1
    end_page = row["end_page"]
    src_lang = row["src_lang"]
    
    for page_num in range(start_page, end_page):
        # Extract the text from the PDF page
        page = reader.pages[page_num]
        page_text = page.extract_text()
        page_text = re.sub(r' {2,}', ' ', page_text)

        # Initialize the chunks
        chunks = []
        current_chunk = ""
        current_chunk_length = 0

        for line in page_text.split(". \n"):
            # Check if the line is not empty
            if line.strip() != "":
                # Translate the line to English
                if src_lang != "en":
                    result = translator_client.translate(
                        values=line.strip(),
                        target_language="en",
                        source_language=src_lang,
                    )
                    tranlated_line = result["translatedText"]
                else:
                    tranlated_line = line
                
                # Add the translated line to the current chunk
                current_chunk += tranlated_line + ".\n"
                current_chunk_length += len(tranlated_line)
            else:
                continue

            # Check if current chunk has reached the character limit
            if current_chunk_length > char_limit:
                # If it does, finalize the current chunk
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_length = 0
        
        # Add the last chunk to the list
        if current_chunk != "":
            if len(chunks) == 0:
                chunks.append(current_chunk.strip())
            else:
                chunks[-1] += current_chunk.strip()
        
        # Add the chunks to the chunk data
        for chunk in chunks:
            chunk_data["id"].append(f"chunk-{uuid.uuid4().hex}")
            chunk_data["content"].append(chunk)
            chunk_data["page"].append(page_num + 1)
            chunk_data["knowledge_id"].append(row["id"])
            chunk_data["title"].append(row["title"])
            chunk_data["author"].append(row["author"])
            chunk_data["organization"].append(row["organization"])
            chunk_data["published_year"].append(row["published_year"])
    
    # Save the chunk data every 10 chunks
    if (index + 1) % 10 == 0:
        chunk_df = pd.DataFrame(chunk_data)
        chunk_df.to_csv(CHUNK_DATA_PATH, index=False)

# Save the chunk data
chunk_df = pd.DataFrame(chunk_data)
chunk_df.to_csv(CHUNK_DATA_PATH, index=False)