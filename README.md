# Hybrid Markov-Fuzzy QoS-QoE Mapping (synthetic-data experiment)

## Setup (VS Code)
1. Open this folder in VS Code (File > Open Folder).
2. Terminal > New Terminal, then:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > `.venv`

## Run
- Full experiment: `python main.py --out results` (or Run and Debug > "Run full experiment")
- Quick check: `python main.py --runs 4 --length 300 --out results_quick`
- Tests: `pytest`

## Layout
- `main.py` - runs the whole experiment
- `src/config.py` - parameters, anchors, weights, state thresholds
- `src/data_generation.py` - synthetic QoS scenarios
- `src/reference_qoe.py` - independent ground-truth QoE (E-model + IQX)
- `src/dataset.py` - dataset assembly and state assignment
- `src/markov_model.py` - transition matrix M, state values V
- `src/fuzzy_system.py` - Mamdani fuzzy inference and rule base
- `src/baselines.py` - linear and IQX baselines
- `src/hybrid.py` - Markov -> Fuzzy pipeline
- `src/metrics.py` - RMSE, MAE, Pearson, Spearman
- `src/plots.py` - figures
- `tests/` - smoke tests
- `results/` - generated tables and figures
# QoS-QoE-hybrid-framework
