# Cloud VM CPU Forecasting

Gradio app comparing three CPU-utilization forecasters — **Naive**, **LSTM**,
and **Temporal Fusion Transformer (TFT)** — trained on Azure Public Dataset
v2 traces.

- Shows a sample forecast automatically on page load — no upload required.
- Lets you pick from a few bundled example VM series, or upload your own
  CSV (`timestamp`, `cpu_util`[, `vm_id`], at least 168 hourly rows).
- Choose a single model or "All" to compare them side by side.
- TFT also shows its native P10–P90 uncertainty band.

## Files

- `app.py` — the Gradio app.
- `preprocessing.py` — feature pipeline shared with the training notebook.
- `examples/*.csv` — bundled sample series used for the no-upload demo.
- `requirements.txt` — pinned dependencies (match the versions the models
  were trained/exported with).

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Model weights (`lstm_model.pt`, `tft_model.ckpt`, `model_config.json`,
`tft_training_dataset.pkl`) are pulled at startup from the Hugging Face Hub
model repo referenced by `HF_MODEL_REPO` in `app.py`.

Full deployment walkthrough: see the accompanying PDF guide.
