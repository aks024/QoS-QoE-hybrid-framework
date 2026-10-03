"""Dataset construction: combines scenarios, reference QoE, composite score and QoS states."""
import numpy as np
import pandas as pd

from src.config import *
from src.data_generation import severity, qos_from_severity
from src.reference_qoe import reference_qoe


def badness(df):
    out = {}
    for p in PARAMS:
        out[p] = np.clip((df[p].values - GOOD[p]) / (BAD[p] - GOOD[p]), 0, 1)
    return out


def composite_score(df):
    b = badness(df)
    return sum(WEIGHTS[p] * b[p] for p in PARAMS)


def build_dataset(n_runs, length, noise, seed):
    frames = []
    for si, sc in enumerate(SCENARIOS):
        for r in range(n_runs):
            rng = np.random.default_rng(seed + 1000 * si + r)
            s = severity(sc, length, rng)
            df = qos_from_severity(s, rng)
            df["scenario"], df["run"], df["t"], df["severity"] = sc, r, np.arange(length), s
            df["qoe_ref"] = reference_qoe(df)
            df["qoe"] = np.clip(df["qoe_ref"] + rng.normal(0, noise, length), 1, 5)  # rating noise
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["composite"] = composite_score(df)
    g = df.groupby(["scenario", "run"])
    df["y_next"] = g["qoe"].shift(-1)  # target: QoE at t+1
    return df


def add_states(df, edges):
    df = df.copy()
    df["state"] = np.digitize(df["composite"].values, edges)
    df["state_next"] = df.groupby(["scenario", "run"])["state"].shift(-1)
    return df


