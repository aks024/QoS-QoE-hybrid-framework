"""Synthetic QoS scenario generator (Gilbert-Elliott loss, AR(1) latency, severity profiles)."""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
def ar1(rng, n, phi=0.9, sd=1.0):
    x = np.zeros(n)
    e = rng.normal(0, sd, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def severity(name, n, rng):
    """Hidden network 'severity' s(t) in [0,1] (0 = perfect, 1 = terrible)."""
    t = np.arange(n)
    if name == "stable":
        s = 0.06 + 0.02 * ar1(rng, n, 0.95, 0.3)
    elif name == "gradual_congestion":
        s = np.linspace(0.05, 0.85, n) + 0.02 * ar1(rng, n, 0.95, 0.3)
    elif name == "sudden_drop":
        t0 = int(n * rng.uniform(0.30, 0.45))
        t1 = int(n * rng.uniform(0.65, 0.80))
        s = np.full(n, 0.08)
        s[t0:t1] = 0.90
        s = s + 0.02 * ar1(rng, n, 0.95, 0.3)
    elif name == "bursty_loss":
        s = np.full(n, 0.15)
        i = 0
        while i < n:
            if rng.random() < 0.03:
                d = rng.geometric(0.1)
                s[i:i + d] = 0.70
                i += d
            else:
                i += 1
        s = s + 0.02 * ar1(rng, n, 0.9, 0.3)
    elif name == "fluctuating":
        ph = rng.uniform(0, 2 * np.pi)
        period = rng.uniform(80, 140)
        s = 0.45 + 0.30 * np.sin(2 * np.pi * t / period + ph) + 0.03 * ar1(rng, n, 0.9, 0.5)
    elif name == "recovery":
        s = 0.10 + 0.80 * np.exp(-t / (n / 4)) + 0.02 * ar1(rng, n, 0.95, 0.3)
    else:
        raise ValueError(name)
    return np.clip(s, 0, 1)


def qos_from_severity(s, rng):
    n = len(s)
    bw = 9.0 * (1 - 0.92 * s) * np.exp(rng.normal(0, 0.06, n))  # available bandwidth, Mbps
    lat = 30 + 350 * s ** 1.3 + ar1(rng, n, 0.8, 1.0) * (6 + 30 * s)  # latency, ms
    lat = np.clip(lat, 10, 900)

    # Gilbert-Elliott bursty packet loss: good/bad channel state, transition probs depend on s
    bad = np.zeros(n, dtype=bool)
    state = False
    for t in range(n):
        p_gb = 0.01 + 0.25 * s[t]
        p_bg = 0.50 - 0.35 * s[t]
        if state:
            state = rng.random() >= p_bg
        else:
            state = rng.random() < p_gb
        bad[t] = state
    loss = np.where(bad,
                    3 + 12 * s + rng.normal(0, 1, n),
                    np.abs(rng.normal(0.15 + 2 * s ** 2, 0.10)))
    loss = np.clip(loss, 0, 40)  # %

    err = np.clip(0.02 + 0.9 * s ** 1.5 + np.abs(rng.normal(0, 0.04, n)), 0, 3)  # %
    thr = np.minimum(2.5, 0.8 * bw) * (1 - loss / 100) * np.exp(rng.normal(0, 0.04, n))  # Mbps
    return pd.DataFrame(dict(throughput=thr, bandwidth=bw, latency=lat, loss=loss, error=err))


