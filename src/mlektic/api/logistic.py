"""Public API for logistic-regression visualization."""

from __future__ import annotations

from ..services.logistic_history import fit_history_logistic
from ..visualization.logistic.router import build_logistic_figure
from ..visualization.theme import attach_highlight

def visualize_logistic(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto",
    show_loss=True,
    title=None,
    smooth="ema",
    smooth_beta=0.85,
    strict_loss=False,
    baseline="prior",
    display_space="original",
    dec=4,
    frame_duration=80,
    theme=None,
):
    """Visualize logistic regression learning process over steps."""
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
        theme=theme,
    )

    return attach_highlight(fig, theme=theme)


__all__ = ["visualize_logistic", "fit_history_logistic", "build_logistic_figure"]
