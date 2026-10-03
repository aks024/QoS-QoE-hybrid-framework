"""Figures for the paper."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import *
from src.fuzzy_system import trap, OUT_MF


def plot_scenarios(df, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for ax, sc in zip(axes.ravel(), SCENARIOS):
        d = df[(df.scenario == sc) & (df.run == 0)]
        ax.plot(d.t, d.qoe, color="tab:green", lw=1, label="QoE (MOS)")
        ax.set_ylim(1, 5)
        ax.set_title(sc)
        ax2 = ax.twinx()
        ax2.plot(d.t, d.latency, color="tab:red", lw=0.7, alpha=0.6, label="latency (ms)")
        ax2.plot(d.t, d.loss * 20, color="tab:blue", lw=0.7, alpha=0.6, label="loss x20 (%)")
        if sc == SCENARIOS[0]:
            ax.legend(loc="lower left", fontsize=7)
            ax2.legend(loc="lower right", fontsize=7)
    fig.supxlabel("time (s)")
    fig.suptitle("Simulated scenarios (run 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_matrix(markov, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(markov.M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(markov.k))
    ax.set_yticks(range(markov.k))
    names = STATE_NAMES_5 if markov.k == 5 else [f"S{i + 1}" for i in range(markov.k)]
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    for i in range(markov.k):
        for j in range(markov.k):
            ax.text(j, i, f"{markov.M[i, j]:.2f}", ha="center", va="center",
                    color="white" if markov.M[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    ax.set_title("Estimated transition matrix M")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_membership(fuzzy, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    rng_max = dict(latency=600, loss=12, throughput=3.0, bandwidth=10, error=1.5)
    for ax, (i, p) in zip(axes.ravel(), enumerate(PARAMS)):
        x = np.linspace(0, rng_max[p], 400)
        for t, abc in fuzzy.mf[p].items():
            ax.plot(x, trap(x, *abc), label=t)
        ax.set_title(f"{p} ({UNITS[p]})")
        ax.legend(fontsize=7)
    ax = axes.ravel()[5]
    for k, v in OUT_MF.items():
        ax.plot(fuzzy.u, trap(fuzzy.u, *v), label=k)
    ax.set_title("QoE output (MOS)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_timeseries(test, preds, path):
    show = ["Fuzzy-only", "Markov-only", "Random Forest", "Hybrid (proposed)"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    first_test_run = test["run"].min()
    for ax, sc in zip(axes, ["sudden_drop", "fluctuating"]):
        m = ((test.scenario == sc) & (test.run == first_test_run)).values
        t = test.loc[m, "t"].values + 1
        ax.plot(t, test.loc[m, "y_next"], color="k", lw=1.2, label="Ground-truth QoE(t+1)")
        for name in show:
            ax.plot(t, np.asarray(preds[name])[m], lw=1, label=name)
        ax.set_title(sc)
        ax.set_ylabel("MOS")
        ax.set_ylim(1, 5)
        ax.legend(fontsize=7, ncol=3)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scenario_bars(tab, path):
    ax = tab.plot(kind="bar", figsize=(12, 5), width=0.85)
    ax.set_ylabel("RMSE (MOS)")
    ax.set_title("RMSE per scenario (test runs)")
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


