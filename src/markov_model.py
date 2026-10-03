"""Markov chain QoS predictor: transition matrix M, state values V, state mean QoE."""
import numpy as np

from src.config import PARAMS


class MarkovQoS:
    def __init__(self, k=5, alpha=1.0):
        self.k, self.alpha = k, alpha

    def fit(self, train):
        v = train.dropna(subset=["state_next"])
        C = np.zeros((self.k, self.k))
        np.add.at(C, (v["state"].astype(int).values, v["state_next"].astype(int).values), 1)
        self.counts = C
        C = C + self.alpha  # Laplace smoothing
        self.M = C / C.sum(1, keepdims=True)  # row-stochastic transition matrix
        V = np.zeros((self.k, len(PARAMS)))
        mu = np.zeros(self.k)
        for s in range(self.k):
            sub = train[train["state"] == s]
            if len(sub) == 0:
                print(f"  [warn] state {s} empty in training data; using global mean")
                sub = train
            V[s] = sub[PARAMS].mean().values  # representative QoS values of each state
            mu[s] = sub["qoe"].mean()  # mean QoE of each state (for Markov-only)
        self.V, self.mu = V, mu
        return self

    def expected_params(self, states):  # X_hat(t+1) = Q_t M V
        return self.M[np.asarray(states, int)] @ self.V

    def expected_qoe(self, states):  # Markov-only baseline
        return self.M[np.asarray(states, int)] @ self.mu


