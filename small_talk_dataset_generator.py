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
    RAFT_SMALL_TALK_DATASET_PATH
)

genai.configure(api_key=GCP_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

prompt = """You are an AI designed to generate greetings and farewells questions and answers for a healthcare chatbot.

You are given the following role-play scenario:
```
You are Mbah-AI, an AI assistant designed to help healthcare professionals answering their questions based on the provided context. You are still permitted to answer greetings and farewells without using the provided context. Other than greetings and farewells, if the information you are looking for is not present in the provided context, say exactly "I'm sorry, Mbah-AI can't answer that question. Is there anything else that I can help? You can ask me anything related to healthcare information."
```

You are also given the following previously generated questions and answers:
```
{previous_questions}
```

Task: Generate 10 unique greetings and farewells questions and answers based on the given role-play scenario.

Notes:
    1. Just use English for the questions and answers.
    2. Each question must be unique.
    3. You must generate greetings and farewells questions and answers that haven't been generated before based on the previous generated questions and answers provided.

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
        "answer": "the answer"
    },
    ... (other questions and answers)
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

previous_questions = []

max_retries = 3
retry_count = 0

for i in tqdm.tqdm(range(10)):
    while retry_count < max_retries:
        result = model.generate_content(
            contents=prompt.format(
                previous_questions=previous_questions if len(previous_questions) != 0 else "No questions and answers are generated yet.",
                output_format=output_format
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
        retry_count = 0
        continue

    for qa in structured_result:
        raft_data["id"].append(f"question-{uuid.uuid4().hex}")
        raft_data["question"].append(qa["question"])
        raft_data["category"].append("SMALL_TALK")

        for j in range(5):
            random_index = random.randint(0, len(chunk_df) - 1)
            raft_data[f"knowledge_{j + 1}"].append({
                "id": chunk_df["id"].iloc[random_index],
                "content": chunk_df["content"].iloc[random_index]
            })

        raft_data["true_knowledge"].append([])
        raft_data["num_true_knowledge"].append(0)
        raft_data["expected_answer"].append(qa["answer"])

    raft_df = pd.DataFrame(raft_data)
    raft_df.to_csv(RAFT_SMALL_TALK_DATASET_PATH, index=False)