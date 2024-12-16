import os

GCP_API_KEY = os.getenv("GCP_API_KEY")
GCP_CRED_PATH = "credentials/application_default_credentials.json"

GEMINI_MODEL_NAME = "gemini-1.5-pro-latest"

KNOWLEDGE_DATA_PATH = "data/knowledge_data.csv"
CHUNK_DATA_PATH = "data/chunk_data.csv"
PROMPT_INJECTION_DATA_PATH = "data/prompt_injection_data.csv"

RAFT_POSITIVE_DATASET_PATH = "data/raft_positive_dataset.csv"
RAFT_NEGATIVE_DATASET_PATH = "data/raft_negative_dataset.csv"
RAFT_SMALL_TALK_DATASET_PATH = "data/raft_small_talk_dataset.csv"
RAFT_TRAIN_DATASET_PATH = "data/raft_train_dataset.csv"
RAFT_TEST_DATASET_PATH = "data/raft_test_dataset.csv"