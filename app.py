import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from preprocessing import prepare_csv

# ---- Point this at YOUR Hugging Face Hub model repo (Section 8.2 of the guide) ----
HF_MODEL_REPO = "your-username/cloud-forecast-models"

# ---- Bundled example CSVs shipped in this repo (examples/*.csv) ----
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
EXAMPLE_FILES = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.csv")))
EXAMPLE_CHOICES = [os.path.basename(p) for p in EXAMPLE_FILES]
EXAMPLE_LOOKUP = {os.path.basename(p): p for p in EXAMPLE_FILES}
DEFAULT_EXAMPLE = EXAMPLE_CHOICES[0] if EXAMPLE_CHOICES else None

# ---- Load config once at startup ----
config_path = hf_hub_download(HF_MODEL_REPO, "model_config.json")
with open(config_path) as f:
    CFG = json.load(f)

FEATURE_COLS = CFG["FEATURE_COLS"]
TARGET_COL = CFG["TARGET_COL"]
HORIZONS = CFG["HORIZONS"]
LOOKBACK = CFG["LOOKBACK"]
VALUE_COLS = CFG["VALUE_COLS"]
MAX_PREDICTION_LENGTH = CFG["MAX_PREDICTION_LENGTH"]
# REFERENCE_VM_ID: a real vm_id from the training set. TFT's GroupNormalizer
# only has fitted scale statistics for vm_ids it saw during training, so a
# new series (uploaded or a bundled example) is mapped onto this one purely
# so the normalizer has something to look up. See the guide's "TFT gotcha".
REFERENCE_VM_ID = CFG["REFERENCE_VM_ID"]

UNKNOWN_REALS = VALUE_COLS + [
    "cpu_lag1", "cpu_lag24", "cpu_roll_mean_6", "cpu_roll_mean_24", "cpu_roll_std_24",
]
KNOWN_REALS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


# ---- Model class (copied verbatim from the training notebook) ----
class MultiHorizonLSTM(nn.Module):
    def __init__(self, n_features=len(FEATURE_COLS), hidden=128, n_layers=2,
                 n_horizons=len(HORIZONS), dropout=0.2):
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


# ---- Load LSTM ----
lstm_path = hf_hub_download(HF_MODEL_REPO, "lstm_model.pt")
lstm_model = MultiHorizonLSTM()
lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
lstm_model.eval()

# ---- Load TFT + the training TimeSeriesDataSet it needs for inference ----
tft_path = hf_hub_download(HF_MODEL_REPO, "tft_model.ckpt")
tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_path, map_location="cpu")
tft_model.eval()

tft_training_path = hf_hub_download(HF_MODEL_REPO, "tft_training_dataset.pkl")
tft_training = TimeSeriesDataSet.load(tft_training_path)


# ---- Forecasters ----
def naive_forecast(series, horizons):
    last = float(series[-1])
    return {h: last for h in horizons}


def run_lstm(feat_df):
    x = feat_df[FEATURE_COLS].values[-LOOKBACK:].astype(np.float32)
    with torch.no_grad():
        pred = lstm_model(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
    return {h: float(pred[i]) for i, h in enumerate(HORIZONS)}


def _future_covariate_rows(feat_df, n_steps):
    """Build the decoder-side rows TFT needs: known covariates for the future
    timestamps, with unknown-real columns placeholder-filled (they're only
    used by the encoder, never fed to the decoder by a TFT)."""
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
    df["vm_id"] = REFERENCE_VM_ID  # map onto a vm_id the fitted normalizer knows
    df = df.iloc[-LOOKBACK:].reset_index(drop=True)
    # re-base time_idx to whatever the reference series' own indexing needs
    df["time_idx"] = np.arange(len(df))

    future = _future_covariate_rows(df, MAX_PREDICTION_LENGTH)
    combined = pd.concat([df, future], ignore_index=True)

    pred_dataset = TimeSeriesDataSet.from_dataset(
        tft_training, combined, predict=True, stop_randomization=True,
    )
    pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        raw = tft_model.predict(pred_loader, mode="quantiles")  # [1, horizon, n_quantiles]

    p10, p50, p90 = raw[0, :, 0].numpy(), raw[0, :, 1].numpy(), raw[0, :, 2].numpy()
    return {
        h: {"pred": float(p50[h - 1]), "lower": float(p10[h - 1]), "upper": float(p90[h - 1])}
        for h in HORIZONS
    }


def _resolve_source(file, example_name):
    """Decide which CSV to read: an upload always wins over the sample picker,
    so users can freely try a sample and then switch to their own data."""
    if file is not None:
        return file.name, f"your upload ({os.path.basename(file.name)})"
    if example_name and example_name in EXAMPLE_LOOKUP:
        return EXAMPLE_LOOKUP[example_name], f"sample series ({example_name})"
    return None, None


def predict(file, example_name, model_choice, horizon):
    horizon = int(horizon)
    source_path, source_label = _resolve_source(file, example_name)

    if source_path is None:
        return None, "Upload a CSV or pick a sample series from the dropdown first."

    raw_df = pd.read_csv(source_path)
    feat_df = prepare_csv(raw_df, VALUE_COLS)

    if len(feat_df) < LOOKBACK:
        return None, (
            f"Need at least {LOOKBACK} hours of history after feature engineering; "
            f"got {len(feat_df)}. Try a longer CSV or one of the bundled samples."
        )

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

    lines = [f"Source: {source_label}"]
    for name, r in results.items():
        if "lower" in r:
            lines.append(f"{name}: {r['pred']:.2f}%  (P10-P90: {r['lower']:.2f}-{r['upper']:.2f}%)")
        else:
            lines.append(f"{name}: {r['pred']:.2f}%")
    summary = "\n".join(lines)

    return fig, summary


# ---------------------------------------------------------------------------
# UI — Blocks (not a plain Interface) so we can auto-run a sample on page
# load and let a bundled example coexist with a real file upload.
# ---------------------------------------------------------------------------
with gr.Blocks(title="Cloud VM CPU Forecasting") as demo:
    gr.Markdown(
        "# Cloud VM CPU Forecasting\n"
        "Compare **Naive**, **LSTM**, and **TFT** forecasts for CPU utilization. "
        "A sample forecast is shown below automatically — no upload needed. "
        "Pick a different bundled sample, or upload your own CSV "
        f"(`timestamp`, `cpu_util`[, `vm_id`], needs at least {LOOKBACK} hourly rows) to override it. "
        "First load after inactivity can take 30-60s while the Space wakes up."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload your own CSV (optional)")
            example_dd = gr.Dropdown(
                EXAMPLE_CHOICES, value=DEFAULT_EXAMPLE,
                label="...or pick a bundled sample series",
            )
            model_radio = gr.Radio(
                ["Naive", "LSTM", "TFT", "All"], value="All", label="Model"
            )
            horizon_radio = gr.Radio(
                [str(h) for h in HORIZONS], value=str(HORIZONS[-1]),
                label="Forecast horizon (hours)",
            )
            run_btn = gr.Button("Run forecast", variant="primary")
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Forecast")
            text_out = gr.Textbox(label="Predicted values", lines=5)

    run_btn.click(
        fn=predict,
        inputs=[file_input, example_dd, model_radio, horizon_radio],
        outputs=[plot_out, text_out],
    )
    # Also re-run whenever the sample picker changes, so switching samples
    # feels immediate even without pressing the button.
    example_dd.change(
        fn=predict,
        inputs=[file_input, example_dd, model_radio, horizon_radio],
        outputs=[plot_out, text_out],
    )
    # Show a sample forecast the instant the page loads -- this is what makes
    # the demo useful before anyone uploads a thing.
    demo.load(
        fn=lambda: predict(None, DEFAULT_EXAMPLE, "All", str(HORIZONS[-1])),
        inputs=None,
        outputs=[plot_out, text_out],
    )

if __name__ == "__main__":
    demo.launch()
