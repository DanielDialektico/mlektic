"""API for building history payloads."""

from ..domain.config import LinearHistoryConfig, LogisticHistoryConfig
from ..history.engine import HistoryEngine

def fit_history(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto", 
    smooth=None, 
    smooth_beta=0.85,
    grid_1d_points=250,
    grid_2d_points=40,
    baseline="mean",
    display_space="original",
    metrics=None
) -> dict:
    """Capture linear history."""
    config = LinearHistoryConfig(
        steps=steps,
        mode=mode,
        smooth=smooth,
        smooth_beta=smooth_beta,
        grid_1d_points=grid_1d_points,
        grid_2d_points=grid_2d_points,
        baseline=baseline,
        display_space=display_space,
        metrics=metrics
    )
    engine = HistoryEngine(trained_estimator)
    return engine.capture_linear(X, y, config)

def fit_history_logistic(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto", 
    smooth=None, 
    smooth_beta=0.85,
    grid_1d_points=300,
    grid_2d_points=40,
    baseline="prior",
    display_space="original"
) -> dict:
    """Capture logistic history."""
    config = LogisticHistoryConfig(
        steps=steps,
        mode=mode,
        smooth=smooth,
        smooth_beta=smooth_beta,
        grid_1d_points=grid_1d_points,
        grid_2d_points=grid_2d_points,
        baseline=baseline,
        display_space=display_space
    )
    engine = HistoryEngine(trained_estimator)
    return engine.capture_logistic(X, y, config)
