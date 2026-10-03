import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.config import PARAMS, EDGES_5
from src.dataset import build_dataset, add_states
from src.fuzzy_system import MamdaniFuzzy
from src.markov_model import MarkovQoS


def test_fuzzy_output_range_and_monotonic():
    f = MamdaniFuzzy()
    good = [[2.5, 8.0, 40, 0.1, 0.02]]
    bad = [[0.4, 1.0, 450, 10, 1.2]]
    g, b = f.predict(good)[0], f.predict(bad)[0]
    assert 1 <= b < g <= 5


def test_markov_rows_sum_to_one():
    df = add_states(build_dataset(2, 200, 0.1, 1), EDGES_5)
    tr = df[df.y_next.notna()]
    m = MarkovQoS(5).fit(tr)
    assert np.allclose(m.M.sum(1), 1.0)
    assert m.V.shape == (5, len(PARAMS))


def test_dataset_shape():
    df = build_dataset(2, 100, 0.1, 1)
    assert len(df) == 6 * 2 * 100
    assert df.qoe.between(1, 5).all()
