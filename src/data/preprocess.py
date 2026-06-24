from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

RAW_PATH = Path("data/raw/freshretailnet_train_sample.parquet")

def preprocess(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%d')
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly['hours_sale'].apply(lambda x: list(range(len(x))))
    df_hourly = df_hourly.explode(['hours_sale','hour'], ignore_index=True)
    df_hourly['hours_sale'] = df_hourly['hours_sale'].astype(float)
    
    store_hourly_demand = (
    df_hourly.groupby(["city_id", "store_id", "dt", "hour"], as_index=False)
    .agg(demand=("hours_sale", "sum"))
    )

    store_hourly_demand["dt"] = pd.to_datetime(store_hourly_demand["dt"])

    store_hourly_demand["datetime"] = (
    store_hourly_demand["dt"]
    + pd.to_timedelta(store_hourly_demand["hour"], unit="h")
    )

    store_hourly_demand = store_hourly_demand[
    ["city_id", "store_id", "datetime", "dt", "hour", "demand"]
    ]

    store_hourly_demand = store_hourly_demand.sort_values(
        ["city_id", "store_id", "datetime"]
    ).reset_index(drop=True)

    if store_hourly_demand[["city_id", "store_id", "datetime"]].duplicated().any():
        raise ValueError("Duplicate city/store/datetime values found after preprocessing.")
    
    if not store_hourly_demand["hour"].between(0, 23).all():
        raise ValueError("Hour column contains values outside 0-23.")
    
    return store_hourly_demand

def sliding_window(dataset):
    dataset = dataset.sort_values(["city_id", "store_id", "datetime"]).reset_index(drop=True)

    for i in range(1, 25):
        dataset[f"lag_{i}"] = (
            dataset.groupby(["city_id", "store_id"])["demand"]
            .shift(i)
        )

    dataset = dataset.dropna().reset_index(drop=True)
    return dataset

def main() -> None:
    PROCESSED_PATH = Path("data/processed")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    store_hourly_demand = preprocess(RAW_PATH)
    store_hourly_demand.to_parquet(f'{PROCESSED_PATH}/store_hourly_demand_sample.parquet', index=False)

    rolling = sliding_window(store_hourly_demand)
    rolling.to_parquet(f'{PROCESSED_PATH}/store_rolling_sample.parquet', index=False)

    n_row = store_hourly_demand.shape[0]
    
    print(f"Saved hourly_demand dataset with {n_row} rows to {PROCESSED_PATH}")
    print(store_hourly_demand.head())
    print(f"Saved rolling window dataset to {PROCESSED_PATH}")
    print(rolling.head())

if __name__ == "__main__":
    main()