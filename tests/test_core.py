import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import fit_history


def test_fit_history_sanity():
    """Basic sanity check to ensure fit_history runs without error on simple data."""
    X = np.random.rand(50, 1)
    y = 2.0 * X[:, 0] + 1.0 + np.random.randn(50) * 0.1

    model = SGDRegressor(max_iter=10)
    model.fit(X, y)

    history = fit_history(model, X, y, steps=5)

    assert isinstance(history, dict)
    assert "loss_hist" in history
    assert len(history["loss_hist"]) == 5
    assert history["history_kind"] in ("iterative", "final_interp")


def test_fit_history_pipeline():
    """Sanity check with Pipeline and StandardScaler."""
    X = np.random.rand(50, 2)
    y = X[:, 0] - X[:, 1] + 0.5

    model = Pipeline([("scaler", StandardScaler()), ("sgd", SGDRegressor(max_iter=10))])
    model.fit(X, y)

    history = fit_history(model, X, y, steps=3, display_space="original")

    assert "w_hist" in history
    if history["w_hist"] is not None:
        assert history["w_hist"].shape[0] == 3
