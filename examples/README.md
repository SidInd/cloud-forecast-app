Drop the CSVs exported by the "export sample data" Colab cell here
(sample_vm_1.csv, sample_vm_2.csv, sample_vm_3.csv, ...).

Each file needs at least `LOOKBACK + max(HORIZONS)` hourly rows
(168 + 24 = 192, with some margin) and the columns:
timestamp, cpu_util, vm_id, vm_type

app.py picks these up automatically at startup (alphabetically first =
the default shown on page load) — no code changes needed, just add the
files and push.
