import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PhishingTextDataset(Dataset):
    """
    PyTorch Dataset for text-based phishing detection.
    CSV must contain columns: text, label (optional: id)
    """

    def __init__(self, csv_path, max_len=128):
        self.data = pd.read_csv(csv_path)

        # 🔥 Fix missing text
        self.data["text"] = self.data["text"].fillna("")
        self.data["text"] = self.data["text"].apply(
            lambda x: x if len(str(x).strip()) > 0 else "[NO_TEXT]"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["text"])

        label = torch.tensor(
            self.data.iloc[idx]["label"], dtype=torch.float
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        sample = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label
        }

        # 🔥 OPTIONAL: include ID if exists
        if "id" in self.data.columns:
            sample["id"] = self.data.iloc[idx]["id"]

        return sample