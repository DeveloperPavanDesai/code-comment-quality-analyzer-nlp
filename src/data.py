"""
Load and prepare CodeSearchNet comment dataset for semantic quality training.
"""
import os
import pandas as pd

from .config import DATA_DIR, DATA_PATH, SAMPLE_SIZE
from .preprocessing import preprocess_text, semantic_score, semantic_label


def load_dataset_from_huggingface(split="train", sample_size=None):
    """Load CodeSearchNet pair dataset from HuggingFace and return a DataFrame."""
    from datasets import load_dataset

    dataset = load_dataset("sentence-transformers/codesearchnet", "pair")
    df = dataset[split].to_pandas()
    df = df[["code", "comment"]].copy()
    df = df.dropna()
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    return df


def prepare_dataframe(df):
    """Add word_count, quality, clean_text, semantic_score, semantic_quality."""
    df = df.copy()
    df["word_count"] = df["comment"].apply(lambda x: len(str(x).split()))
    df["clean_text"] = df["comment"].apply(preprocess_text)
    df["semantic_score"] = df["clean_text"].apply(semantic_score)
    df["semantic_quality"] = df["semantic_score"].apply(semantic_label)
    # Alias for train pipeline
    df["clean_comment"] = df["clean_text"]
    return df


def load_or_create_dataset(sample_size=None, force_download=False):
    """
    Load dataset from CSV if it exists; otherwise download from HuggingFace,
    prepare, and save. Returns DataFrame with clean_comment and semantic_quality.
    """
    size = sample_size or SAMPLE_SIZE
    if not force_download and os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "clean_comment" not in df.columns and "clean_text" in df.columns:
            df["clean_comment"] = df["clean_text"]
        if "semantic_quality" not in df.columns and "clean_text" in df.columns:
            df["semantic_score"] = df["clean_text"].apply(semantic_score)
            df["semantic_quality"] = df["semantic_score"].apply(semantic_label)
        return df

    df = load_dataset_from_huggingface(split="train", sample_size=size)
    df = prepare_dataframe(df)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load_or_create_dataset(sample_size=50_000, force_download=True)
    print(df.shape)
    print(df["semantic_quality"].value_counts())