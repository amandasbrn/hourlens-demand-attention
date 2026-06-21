import torch
from src.models.temporal_attention import LSTMAttentionForecaster


from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.models.temporal_attention import LSTMAttentionForecaster
from src.evaluation.metrics import calculate_regression_metrics

PROCESSED_PATH = Path("data/processed/rolling_sample.parquet")
PRED_PATH = Path("outputs/predictions/lstm_attention_predictions.parquet")
METRICS_PATH = Path("outputs/evaluation/lstm_attention_metrics.csv")
CHECKPOINT_PATH = Path("outputs/checkpoints/lstm_attention.pt")
ATTENTION_PATH = Path("outputs/attention/lstm_attention_weights.parquet")

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

    train_end = int(len(X) * 0.70)
    val_end = int(len(X) * 0.85)

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    val_df = dataset.iloc[train_end:val_end].copy()
    test_df = dataset.iloc[val_end:].copy()

    # normalization

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # X_train_scaled, X_test_scaled = scaler(x_scaler, X_train, X_test)
    # y_train_scaled, y_test_scaled = scaler(y_scaler, y_train, y_test)

    X_train_scaled = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    # convert to tensor
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # create dataloader

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, y_scaler, y_val_scaled, y_test_scaled, val_df, test_df

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    optimizer,
    num_epochs: int,
    checkpoint_path: Path,
    patience: int = 20,
) -> nn.Module:
    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            y_pred, _ = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = evaluate_loss(model, val_loader, loss_fn)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return model

def predict(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions = []
    attention_list = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred, attn_weights = model(X_batch)
            predictions.append(y_pred)
            attention_list.append(attn_weights)

    y_pred_scaled = torch.cat(predictions).numpy()
    attention_weights = torch.cat(attention_list).numpy()

    return y_pred_scaled, attention_weights

def evaluate_loss(model: nn.Module, data_loader: DataLoader, loss_fn) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            y_pred, _ = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    data = read_processed(PROCESSED_PATH)
    train_loader, val_loader, test_loader, y_scaler, y_val_scaled, y_test_scaled, val_df, test_df = prepare_lstm(data)

    ### MODEL TRAINING ###

    model = LSTMAttentionForecaster(
        input_size=1,
        hidden_size=32,
        num_layers=1,
        output_size=1,
    )

    # sample_X = torch.randn(32, 24, 1)

    # prediction, attention_weights = model(sample_X)

    # print(prediction.shape)
    # print(attention_weights.shape)
    # print(attention_weights[0].sum())

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=350,
        checkpoint_path=CHECKPOINT_PATH,
        patience=40,
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved model checkpoint to {CHECKPOINT_PATH}")

    ### PREDICTION ###

    y_pred_scaled, attention_weights = predict(model, test_loader)

    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test_scaled)

    ### EVALUATION ###
    
    metrics = calculate_regression_metrics(y_test_original, y_pred_original)

    print(f"LSTM Attention MAE: {metrics['mae']:.4f}")
    print(f"LSTM Attention RMSE: {metrics['rmse']:.4f}")

    attention_predictions = test_df[["datetime"]].copy()
    attention_predictions["actual"] = y_test_original.flatten()
    attention_predictions["prediction"] = y_pred_original.flatten()
    attention_predictions["model"] = "lstm_attention"

    attention_cols = [f"attn_t_minus_{i}" for i in range(24, 0, -1)]
    attention_df = test_df[["datetime"]].copy()
    attention_df[attention_cols] = attention_weights
    attention_df["model"] = "lstm_attention"

    ATTENTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    attention_df.to_parquet(ATTENTION_PATH, index=False)

    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    attention_predictions.to_parquet(PRED_PATH, index=False)

    metrics_df = pd.DataFrame([
        {
            "model": "lstm_attention",
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
        }
    ])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)

if __name__ == "__main__":
    main()
