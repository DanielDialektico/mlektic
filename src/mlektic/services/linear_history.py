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
    metrics=None,
    max_frames=60,
    frame_step=10,
) -> dict:
    """Construct an auditable linear history payload.

    Incremental estimators are replayed over a clone when resolved by ``mode``;
    other estimators use synthetic interpolation. The returned payload contains
    provenance, source/display timelines, raw/display loss, parameters, grids,
    predictions, and metrics where available. The supplied estimator is not fit
    or mutated.
    """
    config = LinearHistoryConfig(
        steps=steps,
        mode=mode,
        smooth=smooth,
        smooth_beta=smooth_beta,
        grid_1d_points=grid_1d_points,
        grid_2d_points=grid_2d_points,
        baseline=baseline,
        display_space=display_space,
        metrics=metrics,
        max_frames=max_frames,
        frame_step=frame_step,
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
    display_space="original",
    metrics=None,
    max_frames=60,
    frame_step=10,
    multiclass_link="auto",
) -> dict:
    """Construct an auditable binary or multiclass logistic history payload.

    See :func:`fit_history` for replay/interpolation semantics. Fitted class
    order and resolved probability-link semantics remain in the payload.
    """
    config = LogisticHistoryConfig(
        steps=steps,
        mode=mode,
        smooth=smooth,
        smooth_beta=smooth_beta,
        grid_1d_points=grid_1d_points,
        grid_2d_points=grid_2d_points,
        baseline=baseline,
        display_space=display_space,
        metrics=metrics,
        max_frames=max_frames,
        frame_step=frame_step,
        multiclass_link=multiclass_link,
    )
    engine = HistoryEngine(trained_estimator)
    return engine.capture_logistic(X, y, config)
