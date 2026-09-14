#Shared feature pipeline for the Cloud Forecasting app.

import numpy as np
import pandas as pd


def resample_hourly(df: pd.DataFrame, value_cols) -> pd.DataFrame:
    parts = []
    for vm_id, g in df.groupby("vm_id"):
        g = g.set_index("timestamp").sort_index()
        agg = {c: "mean" for c in value_cols}
        res = g[value_cols].resample("1h").agg(agg).interpolate(limit=6)
        for c in ["vm_type", "region", "cloud_provider"]:
            res[c] = g[c].iloc[0]
        res["vm_id"] = vm_id
        parts.append(res.reset_index())
    return pd.concat(parts, ignore_index=True).dropna(subset=value_cols)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["vm_id", "timestamp"]).copy()

    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"] = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_weekend"] = (df["dow"] >= 5).astype(float)
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
    df["time_idx"] = df["time_idx"].astype(int)

    g = df.groupby("vm_id")
    df["cpu_lag1"] = g["cpu_util"].shift(1)
    df["cpu_lag24"] = g["cpu_util"].shift(24)
    df["cpu_roll_mean_6"] = g["cpu_util"].transform(lambda s: s.shift(1).rolling(6).mean())
    df["cpu_roll_mean_24"] = g["cpu_util"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["cpu_roll_std_24"] = g["cpu_util"].transform(lambda s: s.shift(1).rolling(24).std())

    df = df.dropna().reset_index(drop=True)
    return df


def prepare_csv(raw_df: pd.DataFrame, value_cols) -> pd.DataFrame:
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "vm_id" not in df.columns:
        df["vm_id"] = "uploaded-series"
    if "vm_type" not in df.columns:
        df["vm_type"] = "unknown"
    if "region" not in df.columns:
        df["region"] = "user-upload"
    if "cloud_provider" not in df.columns:
        df["cloud_provider"] = "user-upload"

    hourly = resample_hourly(df, value_cols)
    return build_features(hourly)

prepare_uploaded_csv = prepare_csv
