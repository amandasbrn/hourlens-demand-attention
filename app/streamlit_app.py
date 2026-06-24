import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="HourLens", page_icon="📈")

st.title("HourLens: Temporal Attention Demand Forecasting")
st.markdown("### Forecast demand, compare models, and see which past hours mattered most.")
st.info(
    "This dashboard compares several forecasting models for hourly demand. "
    "Lower MAE and RMSE values mean better predictions. "
    "The attention chart shows which past hours the model relied on most."
)

tab1, tab2, tab3 = st.tabs([
    "Model performance",
    "Forecast explorer",
    "Explainable forecast (attention)",
])

METRICS_PATH = Path("outputs/evaluation/store_metrics_summary.csv")
RF_PRED_PATH = Path("outputs/predictions/store_random_forest_predictions.parquet")
LSTM_BASE_PRED_PATH = Path("outputs/predictions/store_lstm_baseline_predictions.parquet")
LSTM_ATTN_PRED_PATH = Path("outputs/predictions/store_lstm_attention_predictions.parquet")
LSTM_ATTN_REG_PRED_PATH = Path("outputs/predictions/store_lstm_attention_reg_predictions.parquet")
ATTENTION_PATH = Path(f"outputs/attention/store_lstm_attention_reg_weights.parquet")

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

rf_pred_df = load_parquet(RF_PRED_PATH)
lstm_base_pred_df = load_parquet(LSTM_BASE_PRED_PATH)
lstm_attn_pred_df = load_parquet(LSTM_ATTN_PRED_PATH)
lstm_attn_reg_pred_df = load_parquet(LSTM_ATTN_REG_PRED_PATH)

MODEL_LABELS = {
    "seasonal_naive_lag_24": "Seasonal naive baseline",
    "store_random_forest": "Random Forest",
    "store_lstm_baseline": "LSTM baseline",
    "store_lstm_attention": "LSTM + Attention",
    "store_lstm_attention_reg": "Regularized LSTM + Attention",
}

prediction_df = pd.concat([rf_pred_df, lstm_base_pred_df, lstm_attn_pred_df, lstm_attn_reg_pred_df])
prediction_df["model_display"] = prediction_df["model"].map(MODEL_LABELS)

with tab1:
    st.header("Summary")
    metrics_df = load_csv(METRICS_PATH)
    metrics_df["model_display"] = metrics_df["model"].map(MODEL_LABELS)
    best_model = metrics_df.sort_values("mae").iloc[0]

    st.metric("best model", best_model['model_display'], width='content')
    col1, col2 = st.columns(2)
    col1.metric("Best MAE", f"{float(best_model['mae']):.4f}")
    col2.metric("Best RMSE", f"{float(best_model['rmse']):.4f}")

    st.dataframe(metrics_df[['model_display','mae','rmse']])

    st.header("Model Comparison")
    metric = st.selectbox("Metric", ["mae", "rmse"])
    metrics_df = metrics_df.sort_values(metric, ascending=False)
    st.bar_chart(metrics_df, x="model_display", y=metric, stack=False, horizontal=True)
    # fig = px.bar(
    # metrics_df.sort_values(metric),
    # y=metric,
    # x="model_display",
    # orientation="h",
    # title=f"Model comparison by {metric.upper()}",
    # labels={
    #     metric: f"{metric.upper()} - lower is better",
    #     "model_display": "Model",
    # })

    with st.expander("What do MAE and RMSE mean?"):
        st.write(
            """
            **MAE** measures the average prediction error.  
            For example, MAE = 4.68 means the model is off by about 4.68 demand units on average.

            **RMSE** also measures prediction error, but it penalizes large mistakes more strongly.  
            Lower values are better for both metrics.
            """
        )
    
    st.subheader("Key takeaway")

    st.success(
    "The LSTM baseline achieved the best accuracy, while the regularized attention model "
    "remained competitive and provides interpretability through past-hour attention weights."
    )

with tab2:
    st.header("Demand forecast")
    selected_store = st.selectbox(
    "Select store",
    sorted(prediction_df["store_id"].unique())
    )

    model_choice = st.selectbox(
        "Select model",
        prediction_df["model_display"].unique()
    )

    model_df = prediction_df[prediction_df['model_display']==model_choice]

    model_df = model_df[model_df["store_id"] == selected_store]

    st.markdown(f"### {model_df['model_display'].iloc[0]} actual vs prediction demand for store {selected_store}.")
    st.caption(
    "A good forecast follows the same pattern as the actual demand line, especially around peaks and drops."
    )

    st.line_chart(model_df, x='datetime', y=['actual','prediction'], color=["#0000FF","#FF0000"])

    model_df["error"] = model_df["prediction"] - model_df["actual"]

    st.markdown(f"### {model_df['model_display'].iloc[0]} prediction error over time")
    st.caption(
    "Positive error means the model over-predicted demand. Negative error means it under-predicted demand.")

    fig_error = px.line(
        model_df,
        x="datetime",
        y="error",
        labels={
            "datetime": "Time",
            "error": "Prediction error",
        },
    )

    st.plotly_chart(fig_error, use_container_width=True)

with tab3:
    st.header("Attention Visual")
    attention_weights_df = load_parquet(ATTENTION_PATH)

    selected_store_attn = st.selectbox(
    "Select store",
    sorted(attention_weights_df["store_id"].unique()), key='attn'
    )

    attention_weights_df = attention_weights_df[
        attention_weights_df["store_id"] == selected_store_attn
    ]

    st.markdown("### Average attention by lag")
    st.caption(
        """
        The attention model looks at the previous 24 hours before making a forecast.
        Each bar shows how much the model relied on that past hour on average.
        """
    )

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

    top_lag = avg_attention.sort_values("avg_attention", ascending=False).iloc[0]

    top_lag_num = int(top_lag["lag_num"])

    st.success(
        f"The model relies most on **t-{top_lag_num}**, "
        f"which means the demand at store {selected_store} from **{top_lag_num} hour(s) before the forecast** "
        "is the most influential on average."
    )

    fig = px.bar(
        avg_attention,
        x="lag_label",
        y="avg_attention",
        labels={
            "lag_label": "Past hour",
            "avg_attention": "Average attention weight",
        },
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Single attention")
    st.caption(
    "For a single forecast, higher bars show which past hours influenced that specific prediction the most."
)

    selected_time = st.selectbox(
    "Select forecast timestamp",
    attention_weights_df["datetime"]
)

    selected_attention = attention_weights_df[
        attention_weights_df["datetime"] == selected_time
    ].iloc[0]

    single_attention = pd.DataFrame({
        "lag": attention_cols,
        "attention_weight": selected_attention[attention_cols].values,
    })

    single_attention["lag_num"] = (
        single_attention["lag"]
        .str.extract(r"minus_(\d+)")
        .astype(int)
    )

    single_attention["lag_label"] = "t-" + single_attention["lag_num"].astype(str)
    single_attention = single_attention.sort_values("lag_num", ascending=False)

    top_lag_single = single_attention.sort_values("attention_weight", ascending=False).iloc[0]

    top_lag_single_num = int(top_lag_single["lag_num"])

    st.success(
        f"Apparently, {selected_time} relies most on **t-{top_lag_single_num}**, "
        f"which means the demand from **{top_lag_single_num} hour(s) before the forecast** "
        "is the most influential on the selected time."
    )

    fig_single = px.bar(
        single_attention,
        x="lag_label",
        y="attention_weight",
        title=f"Attention weights for forecast at {selected_time}",
        labels={
            "lag_label": "Past hour",
            "attention_weight": "Attention weight",
        },
    )

    st.plotly_chart(fig_single, use_container_width=True)