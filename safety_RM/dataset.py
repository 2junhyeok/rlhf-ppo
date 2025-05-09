import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split


def get_splits(seed=42, val_ratio=0.2):
    df = pd.read_json("hf://datasets/MrBananaHuman/kor_ethical_question_answer/train.jsonl", lines=True)
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)

    return train_df, val_df


class EthicalQADataset(Dataset):
    def __init__(self, dataframe, model_name="beomi/kcbert-base", max_length=128):
        self.dataset = dataframe.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        question = item["question"]
        answer = item["answer"]
        reward = item["label"]
        
        text = f"{question} [SEP] {answer}"
        encoding = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "reward": torch.tensor(reward, dtype=torch.float32)
        }