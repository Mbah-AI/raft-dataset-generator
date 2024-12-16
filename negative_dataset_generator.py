import google.generativeai as genai
import json
import pandas as pd
import uuid
import random
import tqdm

from constants import (
    GCP_API_KEY,
    GEMINI_MODEL_NAME,
    CHUNK_DATA_PATH,
    PROMPT_INJECTION_DATA_PATH,
    RAFT_NEGATIVE_DATASET_PATH
)

genai.configure(api_key=GCP_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

prompt = """You are given the following healthcare knowledge references:
```
{chunk_data}
```

You are also given the following previously generated questions:
```
{previous_questions}
```

Task: Make 10 questions completely outside the healthcare domain that can't be answered by the provided references.

Notes:
    1. Just use English for the questions.
    2. Each question must be unique.
    3. You must generate questions that haven't been generated before based on the previous questions provided.

You must provide your response in the following JSON format:
```
[list of 10 questions]
```

Important:
    - Failure to follow the provided JSON format will result in non-acceptance of your response
    - Do not include any additional text or information outside the specified JSON format
"""

chunk_df = pd.read_csv(CHUNK_DATA_PATH)
prompt_injection_df = pd.read_csv(PROMPT_INJECTION_DATA_PATH)

raft_data = {
    "id": [],
    "question": [],
    "category": [],
    "knowledge_1": [],
    "knowledge_2": [],
    "knowledge_3": [],
    "knowledge_4": [],
    "knowledge_5": [],
    "true_knowledge": [],
    "num_true_knowledge": [],
    "expected_answer": []
}

selected_chunk = []
previous_questions = []

max_retries = 3
retry_count = 0

expected_answer = "I'm sorry, Mbah-AI can't answer that question. Is there anything else that I can help? You can ask me anything related to healthcare information."

# Create 1000 negative questions
for i in tqdm.tqdm(range(100)):
    for j in range(5):
        random_index = random.randint(0, len(chunk_df) - 1)
        selected_chunk.append({
            "id": chunk_df["id"].iloc[random_index],
            "content": chunk_df["content"].iloc[random_index]
        })
    
    while retry_count < max_retries:
        result = model.generate_content(
            contents=prompt.format(
                chunk_data=json.dumps(selected_chunk, indent=4),
                previous_questions=previous_questions if len(previous_questions) != 0 else "No questions are generated yet."
            )
        )
        text_result = result.text

        try:
            structured_result = json.loads(text_result.replace("```json", "").replace("```", "").strip())
            previous_questions += structured_result 
            break
        except:
            retry_count += 1
    
    if retry_count == max_retries:
        selected_chunk = []
        retry_count = 0
        continue

    for question in structured_result:
        raft_data["id"].append(f"question-{uuid.uuid4().hex}")
        raft_data["question"].append(question)
        raft_data["category"].append("NEGATIVE")

        for j in range(5):
            random_index = random.randint(0, len(chunk_df) - 1)
            raft_data[f"knowledge_{j + 1}"].append({
                "id": chunk_df["id"].iloc[random_index],
                "content": chunk_df["content"].iloc[random_index]
            })

        raft_data["true_knowledge"].append([])
        raft_data["num_true_knowledge"].append(0)
        raft_data["expected_answer"].append(expected_answer)
    
    selected_chunk = []

    raft_df = pd.DataFrame(raft_data)
    raft_df.to_csv(RAFT_NEGATIVE_DATASET_PATH, index=False)

# Append prompt injection data to the negative dataset
for index, row in prompt_injection_df.iterrows():
    for i in range(5):
        random_index = random.randint(0, len(chunk_df) - 1)
        selected_chunk.append({
            "id": chunk_df["id"].iloc[random_index],
            "content": chunk_df["content"].iloc[random_index]
        })
    
    raft_data["id"].append(f"question-{uuid.uuid4().hex}")
    raft_data["question"].append(row["question"])
    raft_data["category"].append("INJECTION")
    raft_data["knowledge_1"].append({"id": selected_chunk[0]["id"], "content": selected_chunk[0]["content"]})
    raft_data["knowledge_2"].append({"id": selected_chunk[1]["id"], "content": selected_chunk[1]["content"]})
    raft_data["knowledge_3"].append({"id": selected_chunk[2]["id"], "content": selected_chunk[2]["content"]})
    raft_data["knowledge_4"].append({"id": selected_chunk[3]["id"], "content": selected_chunk[3]["content"]})
    raft_data["knowledge_5"].append({"id": selected_chunk[4]["id"], "content": selected_chunk[4]["content"]})
    raft_data["true_knowledge"].append([])
    raft_data["num_true_knowledge"].append(0)
    raft_data["expected_answer"].append(expected_answer)

    selected_chunk = []

raft_df = pd.DataFrame(raft_data)
raft_df.to_csv(RAFT_NEGATIVE_DATASET_PATH, index=False)