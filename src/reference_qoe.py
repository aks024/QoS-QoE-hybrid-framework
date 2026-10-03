"""Independent reference QoE: E-model audio + IQX video + interactivity penalty."""
import numpy as np


def reference_qoe(df):
    d = df["latency"].values
    p_eff = df["loss"].values + df["error"].values  # corrupted packets are discarded too
    thr = df["throughput"].values

    # Audio: simplified ITU-T G.107 E-model (R factor -> MOS)
    Id = 0.024 * d + 0.11 * (d - 177.3) * (d > 177.3)
    Bpl, burst_r = 25.1, 1.5
    Ie_eff = 95.0 * p_eff / (p_eff / burst_r + Bpl)
    R = np.clip(93.2 - Id - Ie_eff, 0, 100)
    mos_a = 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6
    mos_a = np.clip(mos_a, 1, 4.5)

    # Video: IQX-style exponential loss impairment x bitrate saturation
    q_rate = 1 - np.exp(-thr / 0.7)
    mos_v = 1 + 3.6 * q_rate * np.exp(-0.22 * p_eff)

    comb = 0.57 * mos_v + 0.43 * mos_a  # video dominates (cf. Min et al. 2020, ref [30])
    inter = 1.2 * (1 - np.exp(-np.maximum(d - 150, 0) / 250))  # conversational interactivity
    return np.clip(comb - inter, 1, 5)


