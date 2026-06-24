from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "Dingdong-Inc/FreshRetailNet-50K"
RAW_DATA_DIR = Path("data/raw")

def load_hf_split(split: str):
    """
        Load one split of the FreshRetailNet-50K dataset from Hugging Face.
        split: which dataset we want to load (train / eval)
    """
    dataset = load_dataset(DATASET_NAME, split=split)
    return dataset

def sample_dataset(dataset, n_rows: int | None = None):
    """
        return the first n_rows sample from HF dataset
        reason: deterministic & simple
        later: add random sampling with seed
    """
    if n_rows is None:
        return dataset
    
    n_rows = min(n_rows, len(dataset))
    return dataset.select(range(n_rows))

def to_dataframe(dataset) -> pd.DataFrame:
    """
        Convert HF dataset split to pandas dataframe
    """
    return dataset.to_pandas()

def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
        save dataframe as a parquet file
        reason: the dataset contains sequence columns like "hours_sale",
                better for tabular ML pipelines
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

def main() -> None: # meaning the function returns nothing
    split = "train"
    n_rows = 100_000

    dataset = load_hf_split(split=split)
    dataset = sample_dataset(dataset, n_rows=n_rows)
    df = to_dataframe(dataset)

    output_path = RAW_DATA_DIR / f"freshretailnet_{split}_sample.parquet"
    save_dataframe(df, output_path)

    # {len(df):,} = :, is thousand separator
    print(f"Saved {len(df):,} rows, {len(df['city_id'].unique())} cities, {len(df['store_id'].unique())} stores to {output_path}")
    print(f"Rows: {df.shape[0]}, Columns: {list(df.columns)}")
    print(df.head())

if __name__ == "__main__":
    main()