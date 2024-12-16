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
    RAFT_POSITIVE_DATASET_PATH
)

genai.configure(api_key=GCP_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

prompt = """You are given the following healthcare knowledge references:
```
{chunk_data}
```

Task: Make as many questions and answers as possible based on the healthcare knowledge references provided.

Notes:
    1. Generate questions that can be answered based on the provided references only.
    2. If you are given more than 1 reference, then each question should cover all the references.
    3. Make the questions as natural as a doctor's questions.
    4. The answers must be clear, accurate, elaborated, and easy to understand by healthcare workers.
    5. Just use English for the questions and answers.
    6. You can generate 10 questions and answers maximum.
    7. Each question must be unique.
    8. Also list down the reference IDs you used only for each question and answer.

You must provide your response in the following JSON format:
```
{output_format}
```

Important:
    - Failure to follow the provided JSON format will result in non-acceptance of your response
    - Do not include any additional text or information outside the specified JSON format
"""

output_format = """[
    {
        "question": "the question",
        "answer": "the answer",
        "references": [list of reference ids used to generate the question and answer]
    },
    ... (other questions, answers, and references)
]"""

chunk_df = pd.read_csv(CHUNK_DATA_PATH)

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

max_retries = 3
retry_count = 0

num_selected_chunk = random.randint(1, 5)

for index, row in tqdm.tqdm(chunk_df.iterrows(), total=len(chunk_df)):
    selected_chunk.append({
        "id": row["id"],
        "content": row["content"]
    })

    if len(selected_chunk) != num_selected_chunk:
        continue
    else:
        while retry_count < max_retries:
            result = model.generate_content(
                contents=prompt.format(
                    chunk_data=json.dumps(selected_chunk, indent=4),
                    output_format=output_format
                )
            )
            text_result = result.text

            try:
                structured_result = json.loads(text_result.replace("```json", "").replace("```", "").strip())
                break
            except:
                retry_count += 1
        
        if retry_count == max_retries:
            selected_chunk = []
            retry_count = 0
            continue

        if len(selected_chunk) < 5:
            # Fill the rest of the selected chunk with random chunks
            for i in range(5 - len(selected_chunk)):
                random_index = random.randint(index + 1, len(chunk_df) - 1)
                selected_chunk.append({
                    "id": chunk_df["id"].iloc[random_index],
                    "content": chunk_df["content"].iloc[random_index]
                })

        for qa in structured_result:
            raft_data["id"].append(f"question-{uuid.uuid4().hex}")
            raft_data["question"].append(qa["question"])
            raft_data["category"].append("POSITIVE")
            raft_data["knowledge_1"].append({"id": selected_chunk[0]["id"], "content": selected_chunk[0]["content"]})
            raft_data["knowledge_2"].append({"id": selected_chunk[1]["id"], "content": selected_chunk[1]["content"]})
            raft_data["knowledge_3"].append({"id": selected_chunk[2]["id"], "content": selected_chunk[2]["content"]})
            raft_data["knowledge_4"].append({"id": selected_chunk[3]["id"], "content": selected_chunk[3]["content"]})
            raft_data["knowledge_5"].append({"id": selected_chunk[4]["id"], "content": selected_chunk[4]["content"]})
            raft_data["true_knowledge"].append(qa["references"])
            raft_data["num_true_knowledge"].append(len(qa["references"]))
            raft_data["expected_answer"].append(qa["answer"])

        selected_chunk = []
        num_selected_chunk = random.randint(1, 5)
    
    raft_df = pd.DataFrame(raft_data)
    raft_df.to_csv(RAFT_POSITIVE_DATASET_PATH, index=False)