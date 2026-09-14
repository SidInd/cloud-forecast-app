Drop the sample CSVs in this folder.

Each file needs at least `LOOKBACK + max(HORIZONS)` hourly rows
(168 + 24 = 192, with some margin) and the columns:
timestamp, cpu_util, vm_id, vm_type

app.py picks these up automatically at startup (alphabetically first =
the default shown on page load).