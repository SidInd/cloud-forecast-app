# Cloud VM CPU Forecasting

A web application that forecasts short-term CPU utilization for cloud virtual machines, comparing three forecasting approaches, a **Naive baseline**, an **LSTM** multi-horizon regressor, and a **Temporal Fusion Transformer (TFT)**, side by side on the same series.

**Live app:** [https://cloud-forecast-app.streamlit.app/](https://cloud-forecast-app.streamlit.app/) 

## Project Background

Cloud VM CPU utilization is noisy but often follows daily and weekly patterns. Being able to forecast utilization a few hours to a day ahead supports use cases like autoscaling and capacity planning. This project trains and compares three forecasting models of increasing complexity on real cloud VM traces, then packages the best-performing models into a deployed, interactive web app.

## Features

* **Three models, one interface** : Naive (last-value), LSTM (multi-horizon regression), and TFT (probabilistic quantile forecasts with P10/P50/P90 uncertainty bands).  
* **Model comparison mode** : run all three models on the same series and compare predictions directly on one chart.  
* **Multiple forecast horizons** : 1-hour, 6-hour, and 24-hour ahead forecasts.  
* **No upload required to try it** : bundled sample CPU-utilization series let a visitor see a forecast immediately; uploading a custom CSV is optional.  
* **Visual output** : historical series plus each model's point forecast (and TFT's uncertainty band) rendered as a chart, with a plain-text summary alongside it.

## Data

Models were trained on hourly-resampled CPU utilization traces from the Azure Public Dataset (v2), covering roughly 400 virtual machines across multiple VM types, regions, and cloud providers.

**Engineered features** (11 total): `cpu_util`, hour-of-day and day-of-week cyclical encodings (`hour_sin/cos`, `dow_sin/cos`), `is_weekend`, 1-hour and 24-hour lags, and 6-hour/24-hour rolling mean and standard deviation.

**Lookback window:** 168 hours (7 days) of history is used to produce each forecast.

## Models

| Model | Description | 24h-ahead MAE |
| :---- | :---- | :---- |
| Naive | Repeats the last observed value | 1.87 |
| LSTM | 2-layer LSTM, multi-horizon output head (1h / 6h / 24h) | **1.85** (best at 24h) |
| TFT | Temporal Fusion Transformer, per-VM normalization, quantile loss (P10/P50/P90) | 2.46 |

LSTM is the strongest point-forecast performer at the longest horizon tested, which is why all three models are still offered rather than only the "best" one, TFT's uncertainty band and Naive's simplicity/interpretability are useful in their own right.

## Architecture

GitHub (code)  →  Hugging Face Hub (model weights & artifacts)  →  Streamlit Community Cloud (running app)

| Piece | Role |
| :---- | :---- |
| GitHub | Source code and version history |
| Hugging Face Hub | Hosts the trained model artifacts (LFS-backed, no small-file size limit) |
| Streamlit Community Cloud | Runs the live Gradio-style web app, free tier |

> **Note:** the project was originally planned for Hugging Face Spaces. Hugging Face changed its free-tier policy during development to require a paid subscription for Gradio/Docker Spaces on standard CPU hardware, so the live app was moved to Streamlit Community Cloud instead. See the Deployment Guide for details.

## Repository Structure

cloud-forecast-app/

├── streamlit\_app.py       \# Main application

├── preprocessing.py        \# Shared feature-engineering pipeline (training \+ inference)

├── requirements.txt

├── examples/                \# Bundled sample CSVs shown without requiring an upload

└── README.md

Model artifacts (`lstm_model.pt`, `tft_model.ckpt`, `tft_training_dataset.pkl`, `model_config.json`, `model_comparison.csv`) are hosted separately on the Hugging Face Hub and downloaded by the app at startup.

## Running Locally

git clone https://github.com/SidInd/cloud-forecast-app.git

cd cloud-forecast-app

pip install \-r requirements.txt

streamlit run streamlit\_app.py

The app downloads model artifacts from the configured Hugging Face Hub repo on first run.

## Deployment & Usage

See the accompanying **Installation/Deployment Guide** for full setup and deployment instructions, and the **User Guide** for how to use the running application.

## Known Limitations

* TFT's per-VM normalization only has fitted statistics for VMs seen during training; an uploaded series is mapped onto a reference VM ID purely so the normalizer has something to use. This is an approximation, not a defect, a fully general fix would require retraining with a global (rather than per-VM) normalizer.  
* The free hosting tier sleeps after inactivity; the first request after a period of no traffic can take 30-60 seconds.

## Future Enhancements

* Retrain TFT with a global normalizer to remove the reference-VM-ID approximation for genuinely new series.  
* Add more bundled sample series covering a wider variety of VM utilization patterns.  
* Add authentication/rate-limiting if usage grows beyond a demo audience.  
* Add automated retraining on a schedule as more data becomes available.  
* Migrate to paid, always-on hosting if the app needs to avoid cold-start delays.

