"""Mamdani fuzzy inference system (min/max, centroid), implemented from scratch."""
import numpy as np

from src.config import PARAMS

# ----------------------------------------------------------------------------------------
BIG = 1e9
# membership functions: trapezoid (a,b,c,d); triangle = (a,b,b,c)
MF_BASE = {
    "latency": {"Low": (0, 0, 60, 150), "Med": (100, 200, 200, 300), "High": (250, 400, BIG, BIG)},
    "loss": {"Low": (0, 0, 0.5, 1.5), "Med": (1, 3, 3, 6), "High": (4, 8, BIG, BIG)},
    "throughput": {"Low": (0, 0, 0.3, 0.8), "Med": (0.5, 1.2, 1.2, 2.0), "High": (1.5, 2.5, BIG, BIG)},
    "bandwidth": {"Low": (0, 0, 1, 3), "Med": (2, 4, 4, 6), "High": (5, 8, BIG, BIG)},
    "error": {"Low": (0, 0, 0.05, 0.2), "Med": (0.1, 0.4, 0.4, 0.8), "High": (0.5, 1.0, BIG, BIG)},
}
OUT_MF = {  # QoE universe 1..5
    "VeryPoor": (1, 1, 1, 2), "Poor": (1, 2, 2, 3), "Fair": (2, 3, 3, 4),
    "Good": (3, 4, 4, 5), "Excellent": (4, 5, 5, 5),
}
# Rule base: (antecedents, consequent, weight). Uses ALL five inputs; includes penalty rules.
L, P, T, B, E = "latency", "loss", "throughput", "bandwidth", "error"
RULES = [
    ({L: "Low", P: "Low", T: "High"}, "Excellent", 1.0),
    ({L: "Low", P: "Low", T: "High", B: "High", E: "Low"}, "Excellent", 1.0),
    ({L: "Low", P: "Low", T: "Med"}, "Good", 1.0),
    ({L: "Med", P: "Low"}, "Good", 1.0),
    ({L: "Low", P: "Med"}, "Good", 1.0),
    ({E: "Med", P: "Low", L: "Low"}, "Good", 1.0),
    ({L: "Med", P: "Med"}, "Fair", 1.0),
    ({T: "Med", B: "Med", L: "Med", P: "Med"}, "Fair", 1.0),
    ({T: "Low", P: "Low", L: "Low"}, "Fair", 1.0),
    ({L: "Low", P: "Low", B: "Low"}, "Fair", 1.0),
    ({P: "High"}, "Poor", 1.0),
    ({L: "High"}, "Poor", 1.0),
    ({T: "Low", B: "Low"}, "Poor", 1.0),
    ({E: "High"}, "Poor", 1.0),
    ({T: "High", L: "High"}, "Poor", 1.0),
    ({P: "Med", T: "Low"}, "Poor", 1.0),
    ({L: "High", P: "Med"}, "VeryPoor", 1.0),
    ({L: "Med", P: "High"}, "VeryPoor", 1.0),
    ({P: "High", L: "High"}, "VeryPoor", 1.0),
    ({B: "Low", T: "Low", L: "High"}, "VeryPoor", 1.0),
    ({E: "High", P: "Med"}, "VeryPoor", 1.0),
    ({}, "Fair", 0.05),  # weak default so that no input combination is left uncovered
]


def trap(x, a, b, c, d):
    x = np.asarray(x, float)
    left = (x - a) / (b - a) if b > a else (x >= a).astype(float)
    right = (d - x) / (d - c) if d > c else (x <= d).astype(float)
    return np.clip(np.minimum(left, right), 0, 1)


class MamdaniFuzzy:
    def __init__(self, scale=1.0, rules=RULES):
        # 'scale' stretches membership breakpoints (used for the sensitivity analysis)
        self.mf = {p: {t: tuple((v * scale if 0 < v < BIG else v) for v in abc)
                       for t, abc in terms.items()} for p, terms in MF_BASE.items()}
        self.rules = rules
        self.u = np.linspace(1, 5, 201)
        self.out = {k: trap(self.u, *v) for k, v in OUT_MF.items()}

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))  # columns follow PARAMS order
        N = X.shape[0]
        mu = {p: {t: trap(X[:, i], *abc) for t, abc in self.mf[p].items()}
              for i, p in enumerate(PARAMS)}
        agg = np.zeros((N, len(self.u)))
        for ant, out, w in self.rules:
            s = np.full(N, w)
            for p, t in ant.items():
                s = np.minimum(s, mu[p][t])  # AND = min
            agg = np.maximum(agg, np.minimum(s[:, None], self.out[out][None, :]))  # clip + max
        area = agg.sum(1)
        return np.where(area > 1e-9, (agg * self.u).sum(1) / np.maximum(area, 1e-9), 3.0)  # centroid


