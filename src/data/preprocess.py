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
    
    hourly_demand = (
        df_hourly.groupby(['dt','hour'], as_index=False).agg(demand=('hours_sale', 'sum'))
    )

    hourly_demand['dt'] = pd.to_datetime(hourly_demand['dt'])
    hourly_demand['datetime'] = hourly_demand['dt'] + pd.to_timedelta(hourly_demand['hour'], unit='h')

    if hourly_demand["datetime"].duplicated().any():
        raise ValueError("Duplicate datetime values found after preprocessing.")
    
    if not hourly_demand["hour"].between(0, 23).all():
        raise ValueError("Hour column contains values outside 0-23.")

    hourly_demand = hourly_demand[['datetime','dt','hour','demand']]
    hourly_demand = hourly_demand.sort_values("datetime").reset_index(drop=True)
    
    return hourly_demand

def sliding_window(dataset):
    for i in range(1, 25):
        dataset[f'lag_{i}'] = dataset['demand'].shift(i)

    dataset = dataset.dropna().reset_index(drop=True)
    return dataset

def main() -> None:
    PROCESSED_PATH = Path("data/processed")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    hourly_demand = preprocess(RAW_PATH)
    hourly_demand.to_parquet(f'{PROCESSED_PATH}/hourly_demand_sample.parquet', index=False)

    rolling = sliding_window(hourly_demand)
    rolling.to_parquet(f'{PROCESSED_PATH}/rolling_sample.parquet', index=False)

    n_row = hourly_demand.shape[0]
    
    print(f"Saved hourly_demand dataset with {n_row} rows to {PROCESSED_PATH}")
    print(hourly_demand.head())
    print(f"Saved rolling window dataset to {PROCESSED_PATH}")
    print(rolling.head())

if __name__ == "__main__":
    main()