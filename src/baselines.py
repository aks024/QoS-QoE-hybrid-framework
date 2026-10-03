"""Baseline models: fixed linear mapping and fitted IQX."""
import numpy as np
from scipy.optimize import least_squares

from src.config import PARAMS, WEIGHTS


class LinearBaseline:
    """Fixed weighted sum of min-max normalised QoS -> MOS (no learning, no non-linearity)."""

    def fit(self, train):
        self.lo = train[PARAMS].min()
        self.hi = train[PARAMS].max()
        return self

    def predict(self, df):
        q = 0
        for p in PARAMS:
            z = np.clip((df[p].values - self.lo[p]) / (self.hi[p] - self.lo[p] + 1e-12), 0, 1)
            z = z if p in ("throughput", "bandwidth") else 1 - z
            q = q + WEIGHTS[p] * z
        return 1 + 4 * q


class IQXBaseline:
    """QoE = c + a * exp(-(b1*loss + b2*lat/100 + b3*err + b4/(thr+0.2))), fitted by least squares."""

    @staticmethod
    def f(th, X):
        a, c, b1, b2, b3, b4 = th
        z = b1 * X[:, 3] + b2 * X[:, 2] / 100 + b3 * X[:, 4] + b4 / (X[:, 0] + 0.2)
        return c + a * np.exp(-z)

    def fit(self, train):
        X, y = train[PARAMS].values, train["y_next"].values
        lb = [0, 1, 0, 0, 0, 0]
        ub = [5, 5, 20, 20, 20, 20]
        res = least_squares(lambda th: self.f(th, X) - y, x0=[3.5, 1.0, 0.2, 0.3, 0.5, 0.5], bounds=(lb, ub))
        self.th = res.x
        return self

    def predict(self, df):
        return np.clip(self.f(self.th, df[PARAMS].values), 1, 5)


