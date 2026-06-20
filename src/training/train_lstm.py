from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.models.lstm_baseline import LSTMForecaster
from src.evaluation.metrics import calculate_regression_metrics

PROCESSED_PATH = Path("data/processed/rolling_sample.parquet")
PRED_PATH = Path("outputs/predictions/lstm_baseline_predictions.parquet")
METRICS_PATH = Path("outputs/evaluation/lstm_baseline_metrics.csv")
CHECKPOINT_PATH = Path("outputs/checkpoints/lstm_baseline.pt")

def read_processed(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def prepare_lstm(dataset):
    lag_cols = [f"lag_{i}" for i in range(24, 0, -1)]

    # reshape
    X = dataset[lag_cols].values
    y = dataset['demand'].values

    X = X.reshape(X.shape[0], X.shape[1], 1)

    # split data by time

    split_idx = int(len(X) * 0.8)

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    # normalization

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # X_train_scaled, X_test_scaled = scaler(x_scaler, X_train, X_test)
    # y_train_scaled, y_test_scaled = scaler(y_scaler, y_train, y_test)

    X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1,1))
    X_train_scaled = X_train_scaled.reshape(X_train.shape)

    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 1))
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    # convert to tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # create dataloader

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_df = dataset.iloc[split_idx:].copy()

    return train_loader, test_loader, y_scaler, y_test_scaled, test_df

def train_model(model: nn.Module, train_loader: DataLoader, loss_fn, optimizer, num_epochs:int) -> nn.Module:
    for epoch in range(num_epochs): # for each epoch
        model.train() # set model to train mode
        train_loss = 0.0 # set train loss
        for X_batch, y_batch in train_loader: # for each batch
            y_pred = model(X_batch) # predict
            loss = loss_fn(y_pred, y_batch) # compute loss y_pred vs actual (y_batch)

            optimizer.zero_grad() # clear old gradients
            loss.backward() # backprop
            optimizer.step() # update weights

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
    
    return model

def predict(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred)

    y_pred_scaled = torch.cat(predictions).numpy()

    return y_pred_scaled

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    data = read_processed(PROCESSED_PATH)
    train_loader, test_loader, y_scaler, y_test_scaled, test_df = prepare_lstm(data)

    ### MODEL TRAINING ###

    model = LSTMForecaster(
        input_size=1,
        hidden_size=32,
        num_layers=1,
        output_size=1
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=50
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved model checkpoint to {CHECKPOINT_PATH}")

    ### PREDICTION ###

    y_pred_scaled = predict(model, test_loader)

    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test_scaled)

    ### EVALUATION ###
    
    metrics = calculate_regression_metrics(y_test_original, y_pred_original)

    print(f"LSTM MAE: {metrics['mae']:.4f}")
    print(f"LSTM RMSE: {metrics['rmse']:.4f}")

    lstm_predictions = test_df[["datetime"]].copy()
    lstm_predictions["actual"] = y_test_original.flatten()
    lstm_predictions["prediction"] = y_pred_original.flatten()
    lstm_predictions["model"] = "lstm_baseline"

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    lstm_predictions.to_parquet(PRED_PATH, index=False)

    metrics_df = pd.DataFrame([
        {
            "model": "lstm_baseline",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        }
    ])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

if __name__ == "__main__":
    main()
