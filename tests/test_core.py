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


def test_ema_smooths_loss_without_changing_model_geometry():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 1))
    y = 1.7 * X[:, 0] - 0.2 + rng.normal(0, 0.2, size=60)
    model = SGDRegressor(max_iter=20, random_state=9).fit(X, y)

    raw = fit_history(model, X, y, steps=6, smooth=None)
    smoothed = fit_history(model, X, y, steps=6, smooth="ema", smooth_beta=0.8)

    assert np.allclose(smoothed["w_hist"], raw["w_hist"])
    assert np.allclose(smoothed["b_hist"], raw["b_hist"])
    assert np.allclose(smoothed["y_line_hist"], raw["y_line_hist"])
    assert not np.allclose(smoothed["loss_hist"], raw["loss_hist"])
