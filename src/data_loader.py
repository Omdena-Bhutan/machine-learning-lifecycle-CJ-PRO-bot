# src/data_loader.py
import os
import pickle
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer


@dataclass
class PreprocessConfig:
    data_path: str = "data/raw/reviews.csv"
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    train_out: str = "data/processed/train.pkl"
    test_out: str = "data/processed/test.pkl"


class DataPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.cfg = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.data_path)
        required = {"text", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    def split_data(self, df: pd.DataFrame):
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError("Train/Test split resulted in empty set. Check 'split' column values.")

        return train_df, test_df

    def tokenize_texts(self, texts):
        # HuggingFace tokenizer returns dict with input_ids, attention_mask, etc.
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.cfg.max_length,
        )
        return enc

    def build_dataset_dict(self, df: pd.DataFrame):
        enc = self.tokenize_texts(df["text"].astype(str).tolist())
        labels = df["label"].astype(int).tolist()
        enc["labels"] = labels
        return enc

    def save_pickle(self, obj, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def run(self):
        df = self.load_data()
        train_df, test_df = self.split_data(df)

        train_data = self.build_dataset_dict(train_df)
        test_data = self.build_dataset_dict(test_df)

        self.save_pickle(train_data, self.cfg.train_out)
        self.save_pickle(test_data, self.cfg.test_out)

        print(f"[OK] Saved: {self.cfg.train_out} ({len(train_df)} samples)")
        print(f"[OK] Saved: {self.cfg.test_out} ({len(test_df)} samples)")


if __name__ == "__main__":
    cfg = PreprocessConfig()
    DataPreprocessor(cfg).run()
