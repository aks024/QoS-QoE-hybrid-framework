#!/usr/bin/env python3
"""Run the full experiment:  python main.py --out results"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import *
from src.dataset import build_dataset, add_states
from src.fuzzy_system import MamdaniFuzzy
from src.markov_model import MarkovQoS
from src.baselines import LinearBaseline, IQXBaseline
from src.hybrid import hybrid_predict
from src.metrics import metrics, per_sequence_rmse
from src.plots import (plot_scenarios, plot_matrix, plot_membership,
                       plot_timeseries, plot_scenario_bars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="independent runs per scenario")
    ap.add_argument("--length", type=int, default=600, help="samples (seconds) per run")
    ap.add_argument("--noise", type=float, default=0.10, help="std of rating noise on QoE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.7, help="fraction of RUNS used for training")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)

    print("1) Generating data ...")
    df = build_dataset(args.runs, args.length, args.noise, args.seed)
    df.to_csv(f"{args.out}/dataset.csv", index=False)
    print(f"   {len(df)} samples, {df.groupby(['scenario', 'run']).ngroups} sequences")

    n_train_runs = int(round(args.train_frac * args.runs))
    df5 = add_states(df, EDGES_5)
    train = df5[(df5.run < n_train_runs) & df5.y_next.notna()].reset_index(drop=True)
    test = df5[(df5.run >= n_train_runs) & df5.y_next.notna()].reset_index(drop=True)
    print(f"   train runs: 0-{n_train_runs - 1} ({len(train)} pairs) | test runs: {n_train_runs}-{args.runs - 1} ({len(test)} pairs)")
    print("   state occupancy (train):", train.state.value_counts(normalize=True).sort_index().round(3).to_dict())

    print("2) Fitting Markov chain (M, V) ...")
    markov = MarkovQoS(5).fit(train)
    print("   Transition matrix M:\n", np.round(markov.M, 3))
    pd.DataFrame(markov.M, index=STATE_NAMES_5, columns=STATE_NAMES_5).round(4).to_csv(f"{args.out}/transition_matrix.csv")
    st = pd.DataFrame(markov.V, index=STATE_NAMES_5, columns=[f"{p} ({UNITS[p]})" for p in PARAMS])
    st["mean QoE"] = markov.mu
    st.round(3).to_csv(f"{args.out}/state_values_V.csv")
    pd.DataFrame({"boundary": [f"{a}->{b}" for a, b in zip(STATE_NAMES_5[:-1], STATE_NAMES_5[1:])],
                  "composite_badness_threshold": EDGES_5}).to_csv(f"{args.out}/state_thresholds.csv", index=False)

    fuzzy = MamdaniFuzzy()

    print("3) Fitting baselines ...")
    lin = LinearBaseline().fit(train)
    iqx = IQXBaseline().fit(train)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=args.seed)
    rf.fit(train[PARAMS].values, train["y_next"].values)
    sc = StandardScaler().fit(train[PARAMS].values)
    sub = train.sample(min(6000, len(train)), random_state=args.seed)
    svr = SVR(C=3.0, epsilon=0.05).fit(sc.transform(sub[PARAMS].values), sub["y_next"].values)

    print("4) Predicting QoE(t+1) on test runs ...")
    preds = {
        "Linear (fixed)": lin.predict(test),
        "IQX (fitted)": iqx.predict(test),
        "Fuzzy-only": fuzzy.predict(test[PARAMS].values),
        "Markov-only": markov.expected_qoe(test["state"].values),
        "Random Forest": rf.predict(test[PARAMS].values),
        "SVR": svr.predict(sc.transform(test[PARAMS].values)),
        "Hybrid (proposed)": hybrid_predict(markov, fuzzy, test),
        "Persistence (needs measured QoE)": test["qoe"].values,  # reference only
    }
    y = test["y_next"].values

    rows = []
    for name, yh in preds.items():
        m = metrics(y, yh)
        seq = per_sequence_rmse(test, yh)
        m["RMSE_seq_mean"], m["RMSE_seq_std"] = seq.mean(), seq.std()
        rows.append(dict(Model=name, **m))
    overall = pd.DataFrame(rows).set_index("Model").round(4)
    overall.to_csv(f"{args.out}/results_overall.csv")
    print("\n=== Overall test results (predicting QoE at t+1) ===")
    print(overall.to_string())

    by_sc = pd.DataFrame({
        name: [np.sqrt(np.mean((np.asarray(yh)[test.scenario.values == s] - y[test.scenario.values == s]) ** 2))
               for s in SCENARIOS] for name, yh in preds.items()}, index=SCENARIOS).round(3)
    by_sc.to_csv(f"{args.out}/results_by_scenario.csv")
    print("\n=== RMSE per scenario ===")
    print(by_sc.to_string())

    # Ablation is already contained above: Fuzzy-only (no Markov), Markov-only (no fuzzy), Hybrid (both)

    print("\n5) Sensitivity analysis (hybrid RMSE) ...")
    sens = []
    for k in [3, 5, 7]:
        edges = EDGES_5 if k == 5 else list(np.linspace(0.15, 0.70, k - 1))
        d = add_states(df, edges)
        tr = d[(d.run < n_train_runs) & d.y_next.notna()].reset_index(drop=True)
        te = d[(d.run >= n_train_runs) & d.y_next.notna()].reset_index(drop=True)
        mk = MarkovQoS(k).fit(tr)
        sens.append(dict(factor="number of states", value=k,
                         RMSE=metrics(te["y_next"], hybrid_predict(mk, MamdaniFuzzy(), te))["RMSE"]))
    for scale in [0.8, 1.0, 1.2]:
        sens.append(dict(factor="membership scale", value=scale,
                         RMSE=metrics(y, hybrid_predict(markov, MamdaniFuzzy(scale), test))["RMSE"]))
    sens = pd.DataFrame(sens).round(4)
    sens.to_csv(f"{args.out}/sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    print("\n6) Runtime per prediction ...")
    xs = test.sample(300, random_state=1)
    t0 = time.perf_counter()
    for _, r in xs.iterrows():
        hybrid_predict(markov, fuzzy, pd.DataFrame({"state": [int(r["state"])]}))
    ms = (time.perf_counter() - t0) / len(xs) * 1000
    t0 = time.perf_counter()
    for _, r in xs.iterrows():
        fuzzy.predict(r[PARAMS].values.astype(float))
    ms_f = (time.perf_counter() - t0) / len(xs) * 1000
    print(f"   Hybrid: {ms:.3f} ms/prediction | Fuzzy-only: {ms_f:.3f} ms/prediction")
    pd.DataFrame({"model": ["Hybrid", "Fuzzy-only"], "ms_per_prediction": [ms, ms_f]}).round(4).to_csv(f"{args.out}/runtime.csv", index=False)

    print("\n7) Saving figures ...")
    plot_scenarios(df, f"{args.out}/fig_scenarios.png")
    plot_matrix(markov, f"{args.out}/fig_transition_matrix.png")
    plot_membership(fuzzy, f"{args.out}/fig_membership_functions.png")
    plot_timeseries(test, preds, f"{args.out}/fig_timeseries.png")
    plot_scenario_bars(by_sc.drop(columns=["Persistence (needs measured QoE)"]), f"{args.out}/fig_rmse_by_scenario.png")
    print(f"Done. Everything saved in ./{args.out}/")


if __name__ == "__main__":
    main()

