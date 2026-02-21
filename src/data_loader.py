# src/data_loader.py

# WHY: This is the DVC "prepare" stage.
# WHY: It converts raw data into deterministic, training-ready datasets.
# WHAT: Reads raw IMDB CSV (with split column) -> saves train.pkl and test.pkl.

from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/reviews.csv")
OUT_DIR = Path("data/processed")

REQUIRED_COLS = {"text", "label", "split"}


def _validate_schema(df: pd.DataFrame) -> None:
    # WHY: Early, clear failure prevents silent pipeline bugs later.
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Validate split values
    splits = set(df["split"].unique())
    if not {"train", "test"}.issubset(splits):
        raise ValueError(f"Expected split to contain 'train' and 'test'. Got: {sorted(splits)}")


def _clean_text(series: pd.Series) -> pd.Series:
    # WHY: Keep cleaning minimal for transformer models (don't destroy semantics).
    # WHAT: Normalize whitespace + remove newlines.
    return (
        series.astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )


def main() -> None:
    # WHAT: Create output folder if missing
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # WHY: Fail fast if raw data is missing (common CI/CD failure).
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. "
            f"Run: python src/download_data.py"
        )

    # WHAT: Load raw data
    df = pd.read_csv(RAW_PATH)
    _validate_schema(df)

    # WHAT: Basic cleaning + label type normalization
    df["text"] = _clean_text(df["text"])

    # WHY: Ensure labels are integers (0/1) for training code later.
    df["label"] = df["label"].astype(int)

    # WHAT: Deterministic split using existing 'split' column
    train_df = df[df["split"] == "train"][["text", "label"]].reset_index(drop=True)
    test_df = df[df["split"] == "test"][["text", "label"]].reset_index(drop=True)

    # WHY: Sanity checks catch leakage/format issues early.
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Train or test split is empty. train={len(train_df)} test={len(test_df)}")

    # WHAT: Save processed artifacts
    train_path = OUT_DIR / "train.pkl"
    test_path = OUT_DIR / "test.pkl"
    train_df.to_pickle(train_path)
    test_df.to_pickle(test_path)

    print("✅ Prepare stage complete")
    print(f"Saved: {train_path} ({len(train_df)} rows)")
    print(f"Saved: {test_path} ({len(test_df)} rows)")
    print("Columns:", list(train_df.columns))


if __name__ == "__main__":
    main()
