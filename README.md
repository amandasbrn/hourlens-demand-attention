# HourLens: Store-Level Demand Forecasting

Demand forecasting, but make it explainable.

HourLens predicts hourly store demand and shows which past hours the model paid attention to. It is built as an end-to-end AI engineering portfolio project: data pipeline, forecasting models, attention mechanism, evaluation, and Streamlit dashboard.

---

## What this project does

Most forecasting models only give a prediction.

HourLens goes one step further:

- Predicts hourly demand for each store
- Compares multiple forecasting models
- Shows actual vs predicted demand
- Explains which previous hours influenced the forecast
- Turns everything into an interactive dashboard

In simple terms:

> “How much demand should this store expect, and what recent hours shaped that prediction?”

---

## Why it matters

Store-level demand forecasting can help retail teams with:

- staff planning
- inventory preparation
- demand spike monitoring
- store-level operations
- understanding local demand patterns

Instead of only forecasting one global demand number, this project forecasts demand at the store level, which is closer to real business use cases.

---

## Dashboard Preview

The Streamlit dashboard includes:

1. **Model performance**  
   Compare forecasting models using MAE and RMSE.

2. **Forecast explorer**  
   Select a store and inspect actual vs predicted demand.

3. **Attention explanation**  
   See which past hours the attention model relied on most.

---

## Models Compared

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal naive baseline | 1.2778 | 2.1288 |
| Random Forest | 0.8875 | 1.3588 |
| LSTM baseline | 0.8864 | 1.3577 |
| LSTM + Attention | 0.9093 | 1.4059 |
| Regularized LSTM + Attention | 0.8878 | 1.3757 |

---

## Key Takeaway

LSTM baseline wins by a tiny margin, but Random Forest and Regularized LSTM + Attention are right behind.

The attention model stays competitive while adding the extra perk of showing which past hours influenced each forecast.

So the vibe is:

> Best accuracy: LSTM baseline  
> Best explainability: Regularized LSTM + Attention

---

## How attention works here

The model looks at the previous 24 hours before making a forecast.

For example:

- `t-1` means 1 hour before the forecast
- `t-2` means 2 hours before the forecast
- `t-24` means the same hour yesterday

The attention model learns how much each past hour matters.

So instead of being a black box, it can show:

> “I mostly cared about the previous hour, then the hour before that, then a few recent demand patterns.”

---

## Tech Stack

- Python
- pandas
- scikit-learn
- PyTorch
- Streamlit
- Plotly
- Hugging Face Datasets
- Git + GitHub PR workflow

---

## Data

This project uses the FreshRetailNet-50K dataset from Hugging Face.

The raw and processed data files are not committed to GitHub because they are large. The repo keeps only the code, project structure, and small evaluation summaries.

---

## Run the Dashboard

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app/streamlit_app.py
```

---

## ML Pipeline

The project follows this pipeline:

```text
Raw retail data
→ hourly store-level preprocessing
→ lag feature generation
→ model training
→ evaluation
→ attention weight extraction
→ dashboard visualization
```

---

## What I Built

- Store-level hourly demand preprocessing
- Safe grouped lag generation per store
- Seasonal naive baseline
- Random Forest baseline
- LSTM baseline
- LSTM with temporal attention
- Regularized LSTM attention model with:
  - dropout
  - gradient clipping
  - weight decay
  - validation checkpointing
  - early stopping
- Model comparison summary
- Attention weight export
- Streamlit dashboard

---

## Why this is portfolio-worthy

This is not just a notebook model.

It shows an end-to-end AI engineering workflow:

- clean data pipeline
- multiple baselines
- neural network modeling in PyTorch
- interpretability with attention weights
- honest model comparison
- dashboard for non-technical users
- GitHub branch + PR workflow

Basically: model + product + explanation.

---

## Future Work

Possible next upgrades:

- add store embeddings
- add category-level forecasting
- compare with LightGBM or XGBoost
- deploy dashboard to Streamlit Cloud
- add model monitoring page
- add weekly retraining workflow

---

## Project Structure

```text
hourlens-demand-attention/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── outputs/
│   ├── attention/
│   ├── checkpoints/
│   ├── evaluation/
│   ├── figures/
│   └── predictions/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── training/
├── .gitignore
└── README.md

## Project Status

Finished MVP.

Current version supports store-level forecasting and dashboard-based model interpretation.