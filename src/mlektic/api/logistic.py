"""Public API for logistic-regression visualization."""

from __future__ import annotations

from ..services.logistic_history import fit_history_logistic
from ..visualization.logistic.prediction import explain_logistic_prediction
from ..visualization.logistic.router import build_logistic_figure
from ..visualization.theme import attach_highlight, configure_animation


def visualize_logistic(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto",
    show_loss=False,
    title=None,
    smooth="ema",
    smooth_beta=0.85,
    strict_loss=False,
    baseline="prior",
    display_space="original",
    metrics=None,
    dec=4,
    frame_duration=80,
    transition_duration=None,
    max_frames=60,
    frame_step=10,
    max_theta_cols=5,
    multiclass_link="auto",
    theme=None,
):
    """Visualize a logistic-regression learning or interpolation history.

    Binary models use the sigmoid link. For multiclass models,
    ``multiclass_link="auto"`` compares decision scores with ``predict_proba``
    and selects either multinomial Softmax or normalized one-vs-rest sigmoids.

    ``steps`` controls captured states, while ``max_frames`` and ``frame_step``
    bound the rendered history. ``transition_duration`` controls interpolation
    between 2D frames; Plotly 3D traces retain redraw semantics.

    Args:
        trained_estimator: Fitted Scikit-Learn classifier or pipeline.
        X: Training feature matrix.
        y: Training class labels.
        steps: Number of captured states before temporal decimation.
        mode: ``"auto"``, ``"iterative"``, or ``"final_interp"``.
        show_loss: Whether to show loss for a genuine iterative history.
        title: Optional figure title.
        smooth: ``"ema"`` or ``None`` for captured probability and loss series.
        smooth_beta: Exponential moving-average coefficient.
        strict_loss: Raise instead of hiding unavailable empirical loss panels.
        baseline: ``"prior"`` or ``"uniform"`` interpolation baseline.
        display_space: ``"original"`` or ``"scaled"`` parameter space.
        metrics: Built-in metric names or custom metric callables.
        dec: Decimal places used in mathematical substitutions.
        frame_duration: Display time for each native animation frame in
            milliseconds. Dynamic LaTeX equations are discrete layout updates;
            they are redrawn rather than numerically interpolated.
        transition_duration: Visual interpolation time in milliseconds. ``None``
            derives it from ``frame_duration`` and ``0`` disables interpolation.
        max_frames: Maximum uniformly retained frames, or ``None``.
        frame_step: Sampling stride used when ``max_frames`` is ``None``.
        max_theta_cols: Visible class columns before LaTeX truncation.
        multiclass_link: ``"auto"``, ``"softmax"``, or ``"ovr"``.
        theme: Optional visualization theme.

    Returns:
        An animated Plotly figure with mathematically matched definitions.
    """
    hist = fit_history_logistic(
        trained_estimator,
        X,
        y,
        steps=steps,
        mode=mode,
        smooth=smooth,
        smooth_beta=smooth_beta,
        baseline=baseline,
        display_space=display_space,
        metrics=metrics,
        max_frames=max_frames,
        frame_step=frame_step,
        multiclass_link=multiclass_link,
    )

    fig = build_logistic_figure(
        X,
        y,
        history=hist,
        show_loss=show_loss,
        title=title,
        strict_loss=strict_loss,
        dec=dec,
        frame_duration=frame_duration,
        max_theta_cols=max_theta_cols,
        theme=theme,
    )

    configure_animation(fig, frame_duration, transition_duration)
    return attach_highlight(fig, theme=theme)


__all__ = [
    "visualize_logistic",
    "fit_history_logistic",
    "build_logistic_figure",
    "explain_logistic_prediction",
]
