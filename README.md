# HourLens: Store-Level Demand Forecasting with Temporal Attention

**Forecast demand, compare models, and see what the model paid attention to.**

HourLens is an end-to-end AI engineering project for hourly retail demand forecasting. It predicts demand at the **store level** and provides an interactive dashboard to compare model performance, inspect forecasts, and understand which past hours influenced the attention-based model.

---

## Project Overview

Retail demand changes quickly across stores and hours. A global forecast can hide important local patterns, so HourLens focuses on **store-level hourly demand forecasting**.

The project uses historical hourly sales patterns to forecast future demand and visualizes model behavior through temporal attention weights.

The final dashboard answers three questions:

1. Which forecasting model performs best?
2. How close are the predictions to actual demand?
3. Which previous hours did the attention model rely on most?

---

## Why This Project Matters

In retail and e-commerce operations, accurate store-level forecasts can support:

- Store replenishment planning
- Local inventory monitoring
- Staffing and shift planning
- Demand spike detection
- Regional/store-level operational decisions
- Explainable forecasting for business users

Instead of only building a model, this project turns forecasting outputs into a usable Streamlit dashboard.

---

## Dataset

This project uses the `FreshRetailNet-50K` dataset from Hugging Face.

The raw dataset contains retail sales information including:

- `city_id`
- `store_id`
- `product_id`
- `dt`
- `hours_sale`
- category identifiers
- weather and activity-related features

For this version, the project focuses on hourly demand aggregated by:

```text
city_id + store_id + date + hour
```

The model predicts demand using the previous 24 hourly demand values.

---

## Forecasting Task

The forecasting target is:

```text
Predict store-level demand for the next hour.
```

Each training sample uses:

```text
lag_24, lag_23, ..., lag_1
```

to predict:

```text
demand at time t
```

Where:

```text
lag_1  = demand 1 hour before the forecast
lag_24 = demand 24 hours before the forecast
```

Lag features are created **within each store** to avoid cross-store leakage.

---

## Models

The project compares five forecasting approaches:

| Model | Description |
|---|---|
| Seasonal naive baseline | Uses demand from the same hour yesterday |
| Random Forest | Classical machine learning baseline using lag features |
| LSTM baseline | Neural sequence model over the previous 24 hours |
| LSTM + Attention | LSTM with temporal attention over hidden states |
| Regularized LSTM + Attention | Attention model with dropout, gradient clipping, weight decay, validation checkpointing, and early stopping |

---

## Results

Store-level forecasting results:

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal naive baseline | 1.2778 | 2.1288 |
| Random Forest | 0.8875 | 1.3588 |
| LSTM baseline | 0.8864 | 1.3577 |
| LSTM + Attention | 0.9093 | 1.4059 |
| Regularized LSTM + Attention | 0.8878 | 1.3757 |

### Key Takeaway

LSTM baseline wins by a tiny margin, but Random Forest and Regularized LSTM + Attention are right behind. The attention model stays competitive while adding the extra perk of showing which past hours influenced each forecast.

---

## Attention Mechanism

The attention model does not use Transformer-style QKV attention. Instead, it uses learned temporal attention over LSTM hidden states.

The LSTM outputs one hidden state for each of the previous 24 hours:

```text
h_t-24, h_t-23, ..., h_t-1
```

The attention layer learns one weight for each hidden state:

```text
α_t-24, α_t-23, ..., α_t-1
```

Then it creates a weighted context vector:

```text
context = α_t-24 * h_t-24 + α_t-23 * h_t-23 + ... + α_t-1 * h_t-1
```

This allows the dashboard to show which previous hours mattered most for the forecast.

---

## Dashboard

The Streamlit dashboard includes three main sections:

### 1. Model Performance

Shows:

- Metrics summary table
- Best model
- Best MAE
- Best RMSE
- Model comparison chart

### 2. Forecast Explorer

Shows:

- Store-level actual vs predicted demand
- Model selector
- Store selector
- Forecast behavior over time

### 3. Attention Explanation

Shows:

- Average attention by lag
- Attention weights from `t-24` to `t-1`
- Which past hours the attention model relied on most

---

## Example Dashboard Text

The dashboard is designed to be readable for non-technical users:

```text
Compare demand forecasts and see what the model paid attention to.
```

Attention interpretation:

```text
Higher attention means the model relied more on that past hour when making a forecast.
```

---

## Project Structure

```text
hourlens-demand-attention/
├── app/
│   ├── streamlit_app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   ├── baseline_model.ipynb
│   ├── attention_visualisation.ipynb
│   └── metrics_summary.ipynb
│
├── outputs/
│   ├── attention/
│   │   └── .gitkeep
│   ├── checkpoints/
│   │   └── .gitkeep
│   ├── evaluation/
│   │   └── .gitkeep
│   ├── figures/
│   │   └── .gitkeep
│   └── predictions/
│       └── .gitkeep
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── lstm_baseline.py
│   │   └── temporal_attention.py
│   └── training/
│       ├── train_lstm.py
│       └── train_attention_lstm.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline

The project follows this workflow:

```text
1. Load raw retail dataset
2. Aggregate hourly demand by city and store
3. Validate store-level time coverage
4. Create 24-hour lag features per store
5. Train forecasting baselines
6. Train LSTM models
7. Train temporal attention model
8. Save predictions, metrics, checkpoints, and attention weights
9. Build Streamlit dashboard
10. Compare models and interpret attention patterns
```

---

## Data Preprocessing

The raw `hours_sale` column contains 24 hourly values for each row.

The preprocessing step expands this array into hourly rows, then aggregates demand at store level:

```text
city_id + store_id + dt + hour
```

The resulting processed dataset has this format:

```text
city_id | store_id | datetime | dt | hour | demand
```

Lag features are created per store:

```text
lag_1, lag_2, ..., lag_24
```

Grouped lag creation is used to prevent leakage:

```python
df.groupby(["city_id", "store_id"])["demand"].shift(i)
```

---

## Model Training Details

### LSTM Input Shape

The LSTM receives the previous 24 hourly demand values:

```text
(batch_size, sequence_length, input_size)
```

For this project:

```text
(batch_size, 24, 1)
```

### Train / Validation / Test Split

The neural models use a time-based split:

```text
70% train
15% validation
15% test
```

The validation set is used for early stopping and checkpoint selection.

### Regularization

The regularized attention model uses:

- Dropout
- Weight decay
- Gradient clipping
- Early stopping
- Validation-based checkpointing

---

## Evaluation Metrics

The project uses:

### MAE

Mean Absolute Error measures the average prediction error.

```text
Lower MAE = better average prediction accuracy
```

### RMSE

Root Mean Squared Error penalizes larger errors more strongly.

```text
Lower RMSE = fewer large forecasting mistakes
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd hourlens-demand-attention
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run preprocessing

```bash
python src/data/load_data.py
python src/data/preprocess.py
```

### 5. Train models

```bash
python src/training/train_lstm.py
python src/training/train_attention_lstm.py
```

### 6. Run dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Outputs

Generated files are saved locally under `outputs/`.

Examples:

```text
outputs/predictions/
outputs/evaluation/
outputs/attention/
outputs/checkpoints/
outputs/figures/
```

Large generated files are ignored by Git. Small summary files such as evaluation summaries may be tracked when useful for documentation.

---

## Main Output Files

Prediction files:

```text
outputs/predictions/store_random_forest_predictions.parquet
outputs/predictions/store_lstm_baseline_predictions.parquet
outputs/predictions/store_lstm_attention_reg_predictions.parquet
```

Attention files:

```text
outputs/attention/store_lstm_attention_reg_weights.parquet
```

Evaluation files:

```text
outputs/evaluation/store_metrics_summary.csv
```

---

## Engineering Practices

This project was built using a clean AI engineering workflow:

- Feature branches
- Pull requests
- Modular source code
- Reusable metrics functions
- Separate preprocessing, training, evaluation, and dashboard logic
- Git-tracked folder structure with `.gitkeep`
- Generated data and model artifacts ignored with `.gitignore`
- Validation-based model selection
- Dashboard-based model interpretation

---

## What I Learned

This project helped practice:

- End-to-end ML project structure
- Time series feature engineering
- Store-level demand aggregation
- Lag feature creation without leakage
- Random Forest forecasting baseline
- LSTM sequence modeling in PyTorch
- Temporal attention over LSTM hidden states
- Dropout, gradient clipping, weight decay, and early stopping
- Model evaluation with MAE/RMSE
- Streamlit dashboard development
- Interpretable ML storytelling for portfolio presentation

---

## Limitations

- The current model uses only historical demand lag features.
- Store identity is used for filtering and grouping, but not yet as a learned embedding.
- Product/category-level forecasting is not included in this version.
- External features such as weather, discount, holiday, and activity flags are not yet fully integrated into the neural models.
- Attention weights are useful for interpretation, but they should not be treated as perfect causal explanations.

---

## Future Work

Potential extensions:

- Add store embeddings to the LSTM model
- Add weather, discount, holiday, and activity features
- Extend forecasting to category-level or product-level demand
- Compare against stronger forecasting baselines such as LightGBM, XGBoost, GRU, TCN, or Transformer encoder
- Add multi-step forecasting
- Add prediction intervals or uncertainty estimates
- Deploy the Streamlit dashboard
- Add automated training and evaluation pipeline

---

## Project Status

Current version:

```text
Store-level hourly demand forecasting dashboard complete.
```

The project includes:

- Store-level preprocessing
- Forecasting baselines
- LSTM models
- Temporal attention model
- Evaluation summary
- Streamlit dashboard
- Attention visualization

---

## Author

Built as an AI engineering portfolio project.