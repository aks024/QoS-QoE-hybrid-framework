"""Global configuration: parameter names, anchors, weights, state thresholds."""
# 0. Configuration
# ----------------------------------------------------------------------------------------
PARAMS = ["throughput", "bandwidth", "latency", "loss", "error"]  # T, B, L, P, E
UNITS = {"throughput": "Mbps", "bandwidth": "Mbps", "latency": "ms", "loss": "%", "error": "%"}
SCENARIOS = ["stable", "gradual_congestion", "sudden_drop", "bursty_loss", "fluctuating", "recovery"]
STATE_NAMES_5 = ["Excellent", "Good", "Moderate", "Poor", "Very Poor"]

# "good" and "bad" anchor values used to turn each parameter into a 0..1 badness score
# (latency anchors follow ITU-T G.114: <=150 ms acceptable, >400 ms unacceptable)
GOOD = dict(throughput=2.0, bandwidth=6.0, latency=50.0, loss=0.5, error=0.05)
BAD = dict(throughput=0.3, bandwidth=1.0, latency=400.0, loss=8.0, error=1.0)
WEIGHTS = dict(throughput=0.15, bandwidth=0.15, latency=0.30, loss=0.30, error=0.10)
EDGES_5 = [0.15, 0.30, 0.50, 0.70]  # composite-badness thresholds between the 5 states


