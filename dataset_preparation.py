import pandas as pd
from sklearn.model_selection import train_test_split

from constants import (
    RAFT_POSITIVE_DATASET_PATH,
    RAFT_NEGATIVE_DATASET_PATH,
    RAFT_SMALL_TALK_DATASET_PATH,
    RAFT_TRAIN_DATASET_PATH,
    RAFT_TEST_DATASET_PATH
)

raft_positive_df = pd.read_csv(RAFT_POSITIVE_DATASET_PATH)
raft_negative_df = pd.read_csv(RAFT_NEGATIVE_DATASET_PATH)
raft_small_talk_df = pd.read_csv(RAFT_SMALL_TALK_DATASET_PATH)

raft_df = pd.concat([raft_positive_df, raft_negative_df, raft_small_talk_df], ignore_index=True)
raft_df["stratify_on"] = raft_df["category"] + "_" + raft_df["num_true_knowledge"].astype(str)

train_df, test_df = train_test_split(
    raft_df,
    test_size=0.1,
    random_state=42,
    stratify=raft_df["stratify_on"],
    shuffle=True
)

train_df.to_csv(RAFT_TRAIN_DATASET_PATH, index=False)
test_df.to_csv(RAFT_TEST_DATASET_PATH, index=False)