"""Hybrid pipeline: QoS_t --Markov--> X_hat(t+1) --Fuzzy--> QoE(t+1)."""


def hybrid_predict(markov, fuzzy, df):
    return fuzzy.predict(markov.expected_params(df["state"].values))

