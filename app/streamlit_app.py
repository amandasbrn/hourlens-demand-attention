import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

METRICS_PATH = Path("outputs/evaluation/metrics_summary.csv")
RF_PRED_PATH = Path("outputs/predictions/random_forest_predictions.parquet")
LSTM_BASE_PRED_PATH = Path("outputs/predictions/lstm_baseline_predictions.parquet")
LSTM_ATTN_PRED_PATH = Path("outputs/predictions/lstm_attention_predictions.parquet")
LSTM_ATTN_REG_PRED_PATH = Path("outputs/predictions/lstm_attention_reg_predictions.parquet")
ATTENTION_PATH = Path(f"outputs/attention/lstm_attention_reg_weights.parquet")

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

st.title("HourLens: Temporal Attention Demand Forecasting")
st.markdown("This dashboard compares demand forecasting models and visualizes which past hours the attention model uses when predicting future hourly demand.")

st.header("Summary")
metrics_df = load_csv(METRICS_PATH)
st.dataframe(metrics_df)

best_model = metrics_df.sort_values("mae").iloc[0]
st.metric("best model", best_model['model'], width='content')
col1, col2 = st.columns(2)
col1.metric("Best MAE", f"{float(best_model['mae']):.4f}")
col2.metric("Best RMSE", f"{float(best_model['rmse']):.4f}")

st.header("Model Comparison")
metric = st.selectbox("Metric", ["mae", "rmse"])
st.bar_chart(metrics_df, x="model", y=metric, stack=False, horizontal=True)

rf_pred_df = load_parquet(RF_PRED_PATH)
lstm_base_pred_df = load_parquet(LSTM_BASE_PRED_PATH)
lstm_attn_pred_df = load_parquet(LSTM_ATTN_PRED_PATH)
lstm_attn_reg_pred_df = load_parquet(LSTM_ATTN_REG_PRED_PATH)

st.header("Actual vs Prediction demand")
prediction_df = pd.concat([rf_pred_df, lstm_base_pred_df, lstm_attn_pred_df, lstm_attn_reg_pred_df])
model_choice = st.selectbox(
    "Select model",
    prediction_df["model"].unique()
)

model_df = prediction_df[prediction_df['model']==model_choice]

st.line_chart(model_df, x='datetime', y=['actual','prediction'], color=["#0000FF", "#FF0000"])

st.header("Attention Visual")

attention_weights_df = load_parquet(ATTENTION_PATH)

attention_cols = [
    col for col in attention_weights_df.columns 
    if col.startswith("attn_")
]

avg_attention = attention_weights_df[attention_cols].mean().reset_index()
avg_attention.columns = ["lag", "avg_attention"]

avg_attention["lag_num"] = (
    avg_attention["lag"]
    .str.extract(r"minus_(\d+)")
    .astype(int)
)

avg_attention["lag_label"] = "t-" + avg_attention["lag_num"].astype(str)

avg_attention = avg_attention.sort_values("lag_num", ascending=False)

fig = px.bar(
    avg_attention,
    x="lag_label",
    y="avg_attention",
    title="Average attention by lag",
    labels={
        "lag_label": "Past hour",
        "avg_attention": "Average attention weight",
    },
)

st.plotly_chart(fig, use_container_width=True)