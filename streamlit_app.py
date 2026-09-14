import glob
import json
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from preprocessing import prepare_csv

HF_MODEL_REPO = "SidArr/cloud-forecast-models"


EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
EXAMPLE_FILES = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.csv")))
EXAMPLE_CHOICES = [os.path.basename(p) for p in EXAMPLE_FILES]
EXAMPLE_LOOKUP = {os.path.basename(p): p for p in EXAMPLE_FILES}


def _load_tft_cpu_safe(ckpt_path):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(raw["hyper_parameters"])

    model = TemporalFusionTransformer(**hparams)
    model.load_state_dict(raw["state_dict"])

    for module in model.modules():
        if hasattr(module, "_device"):
            try:
                module._device = torch.device("cpu")
            except AttributeError:
                module.__dict__["_device"] = torch.device("cpu")

    model.eval()
    return model


@st.cache_resource(show_spinner="Loading models (first load can take a minute)...")
def load_everything():
    config_path = hf_hub_download(HF_MODEL_REPO, "model_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    class MultiHorizonLSTM(nn.Module):
        def __init__(self, n_features, hidden=128, n_layers=2, n_horizons=3, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_horizons),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    lstm_path = hf_hub_download(HF_MODEL_REPO, "lstm_model.pt")
    lstm_model = MultiHorizonLSTM(n_features=len(cfg["FEATURE_COLS"]), n_horizons=len(cfg["HORIZONS"]))
    lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    lstm_model.eval()

    tft_path = hf_hub_download(HF_MODEL_REPO, "tft_model.ckpt")
    tft_model = _load_tft_cpu_safe(tft_path)

    tft_training_path = hf_hub_download(HF_MODEL_REPO, "tft_training_dataset.pkl")
    tft_training = TimeSeriesDataSet.load(tft_training_path)

    return cfg, lstm_model, tft_model, tft_training


CFG, lstm_model, tft_model, tft_training = load_everything()

FEATURE_COLS = CFG["FEATURE_COLS"]
TARGET_COL = CFG["TARGET_COL"]
HORIZONS = CFG["HORIZONS"]
LOOKBACK = CFG["LOOKBACK"]
VALUE_COLS = CFG["VALUE_COLS"]
MAX_PREDICTION_LENGTH = CFG["MAX_PREDICTION_LENGTH"]
REFERENCE_VM_ID = CFG["REFERENCE_VM_ID"]

UNKNOWN_REALS = VALUE_COLS + [
    "cpu_lag1", "cpu_lag24", "cpu_roll_mean_6", "cpu_roll_mean_24", "cpu_roll_std_24",
]
KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


def naive_forecast(series, horizons):
    last = float(series[-1])
    return {h: last for h in horizons}


def run_lstm(feat_df):
    x = feat_df[FEATURE_COLS].values[-LOOKBACK:].astype(np.float32)
    with torch.no_grad():
        pred = lstm_model(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
    return {h: float(pred[i]) for i, h in enumerate(HORIZONS)}


def _future_covariate_rows(feat_df, n_steps):
    last_ts = feat_df["timestamp"].iloc[-1]
    last_time_idx = int(feat_df["time_idx"].iloc[-1])
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=n_steps, freq="h")

    future = pd.DataFrame({"timestamp": future_ts})
    future["time_idx"] = np.arange(last_time_idx + 1, last_time_idx + 1 + n_steps)
    hour, dow = future["timestamp"].dt.hour, future["timestamp"].dt.dayofweek
    future["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    future["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    future["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    future["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    future["is_weekend"] = (dow >= 5).astype(float)
    for col in UNKNOWN_REALS:
        future[col] = 0.0
    future["vm_id"] = feat_df["vm_id"].iloc[0]
    future["vm_type"] = feat_df["vm_type"].iloc[0]
    return future


def run_tft(feat_df):
    df = feat_df[["timestamp", "time_idx", "vm_type"] + KNOWN_REALS + UNKNOWN_REALS].copy()
    df["vm_id"] = REFERENCE_VM_ID
    df = df.iloc[-LOOKBACK:].reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))

    future = _future_covariate_rows(df, MAX_PREDICTION_LENGTH)
    combined = pd.concat([df, future], ignore_index=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(
        tft_training, combined, predict=True, stop_randomization=True,
    )
    pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        raw = tft_model.predict(pred_loader, mode="quantiles")

    p10, p50, p90 = raw[0, :, 0].numpy(), raw[0, :, 1].numpy(), raw[0, :, 2].numpy()
    return {
        h: {"pred": float(p50[h - 1]), "lower": float(p10[h - 1]), "upper": float(p90[h - 1])}
        for h in HORIZONS
    }


def predict(raw_df, model_choice, horizon, source_label):
    feat_df = prepare_csv(raw_df, VALUE_COLS)

    if len(feat_df) < LOOKBACK:
        st.error(
            f"Need at least {LOOKBACK} hours of history after feature engineering; "
            f"got {len(feat_df)}. Try a longer CSV or a bundled sample."
        )
        return

    series = feat_df[TARGET_COL].values
    results = {}
    if model_choice in ("Naive", "All"):
        results["Naive"] = {"pred": naive_forecast(series, HORIZONS)[horizon]}
    if model_choice in ("LSTM", "All"):
        results["LSTM"] = {"pred": run_lstm(feat_df)[horizon]}
    if model_choice in ("TFT", "All"):
        results["TFT"] = run_tft(feat_df)[horizon]

    fig, ax = plt.subplots(figsize=(9, 4))
    hist_x = feat_df["timestamp"].iloc[-LOOKBACK:]
    ax.plot(hist_x, series[-LOOKBACK:], label="History", color="black")

    target_time = feat_df["timestamp"].iloc[-1] + pd.Timedelta(hours=horizon)
    for name, r in results.items():
        ax.scatter([target_time], [r["pred"]], label=f"{name} ({horizon}h)", zorder=5, s=60)
        if "lower" in r:
            ax.errorbar([target_time], [r["pred"]],
                        yerr=[[r["pred"] - r["lower"]], [r["upper"] - r["pred"]]],
                        fmt="none", ecolor="gray", capsize=4)

    ax.legend()
    ax.set_ylabel("CPU %")
    ax.set_title(f"{horizon}h-ahead forecast — {source_label}")
    plt.tight_layout()
    st.pyplot(fig)

    lines = [f"**Source:** {source_label}"]
    for name, r in results.items():
        if "lower" in r:
            lines.append(f"- **{name}:** {r['pred']:.2f}%  (P10-P90: {r['lower']:.2f}-{r['upper']:.2f}%)")
        else:
            lines.append(f"- **{name}:** {r['pred']:.2f}%")
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cloud VM CPU Forecasting", layout="wide")
st.title("Cloud VM CPU Forecasting")
st.write(
    "Compare **Naive**, **LSTM**, and **TFT** forecasts for CPU utilization. "
    "Pick a bundled sample or upload your own CSV "
    f"(`timestamp`, `cpu_util`[, `vm_id`], needs at least {LOOKBACK} hourly rows). "
    "First load after inactivity can take 30-60s while the app wakes up."
)

col1, col2 = st.columns([1, 2])

with col1:
    uploaded = st.file_uploader("Upload your own CSV (optional)", type="csv")
    example_name = st.selectbox(
        "...or pick a bundled sample series",
        EXAMPLE_CHOICES if EXAMPLE_CHOICES else ["(no samples bundled yet)"],
    )
    model_choice = st.radio("Model", ["Naive", "LSTM", "TFT", "All"], index=3, horizontal=True)
    horizon = st.radio(
        "Forecast horizon (hours)", HORIZONS, index=len(HORIZONS) - 1, horizontal=True,
    )
    run = st.button("Run forecast", type="primary")

with col2:
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        source_label = f"your upload ({uploaded.name})"
        predict(raw_df, model_choice, horizon, source_label)
    elif example_name in EXAMPLE_LOOKUP:
        raw_df = pd.read_csv(EXAMPLE_LOOKUP[example_name])
        source_label = f"sample series ({example_name})"
        predict(raw_df, model_choice, horizon, source_label)
    else:
        st.info("Upload a CSV or add sample CSVs to the examples/ folder to see a forecast here.")
