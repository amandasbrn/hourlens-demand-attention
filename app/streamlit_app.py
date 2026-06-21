import streamlit as st
from pathlib import Path
import pandas as pd

METRICS_PATH = Path("outputs/evaluation/metrics_summary.csv")
RF_PRED_PATH = Path("outputs/predictions/random_forest_predictions.parquet")
LSTM_BASE_PRED_PATH = Path("outputs/predictions/lstm_baseline_predictions.parquet")
LSTM_ATTN_PRED_PATH = Path("outputs/predictions/lstm_attention_predictions.parquet")
LSTM_ATTN_REG_PRED_PATH = Path("outputs/predictions/lstm_attention_reg_predictions.parquet")

st.title("HourLens: Temporal Attention Demand Forecasting")
st.markdown("This dashboard compares demand forecasting models and visualizes which past hours the attention model uses when predicting future hourly demand.")

st.header("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("best model", "lstm_attention_reg", width='content')
col2.metric("MAE", "4.6781")
col3.metric("RMSE", "6.6149")

metrics_df = pd.read_csv(METRICS_PATH)
st.dataframe(metrics_df)

st.header("Model Comparison")
metric = st.selectbox("Metric", ["mae", "rmse"])
st.bar_chart(metrics_df, x="model", y=metric, stack=False, horizontal=True)

rf_pred_df = pd.read_parquet(RF_PRED_PATH)
lstm_base_pred_df = pd.read_parquet(LSTM_BASE_PRED_PATH)
lstm_attn_pred_df = pd.read_parquet(LSTM_ATTN_PRED_PATH)
lstm_attn_reg_pred_df = pd.read_parquet(LSTM_ATTN_REG_PRED_PATH)

st.header("Actual vs Prediction demand")
prediction_df = pd.concat([rf_pred_df, lstm_base_pred_df, lstm_attn_pred_df, lstm_attn_reg_pred_df])
model_choice = st.selectbox(
    "Select model",
    prediction_df["model"].unique()
)

model_df = prediction_df[prediction_df['model']==model_choice]

st.line_chart(model_df, x='datetime', y=['actual','prediction'], color=["#0000FF", "#FF0000"])