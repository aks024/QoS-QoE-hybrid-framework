"""Evaluation metrics."""
import numpy as np
from scipy.stats import pearsonr, spearmanr


def metrics(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    pr = pearsonr(y, yhat)[0] if np.std(yhat) > 1e-9 else np.nan
    sr = spearmanr(y, yhat)[0] if np.std(yhat) > 1e-9 else np.nan
    return dict(RMSE=rmse, MAE=mae, Pearson=float(pr), Spearman=float(sr))


def per_sequence_rmse(test, yhat):
    tmp = test[["scenario", "run", "y_next"]].copy()
    tmp["se"] = (np.asarray(yhat) - tmp["y_next"].values) ** 2
    return tmp.groupby(["scenario", "run"])["se"].mean().pipe(np.sqrt)


