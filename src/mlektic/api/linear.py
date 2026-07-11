"""Public API for linear-regression visualization."""

from __future__ import annotations

from ..services.linear_history import fit_history
from ..visualization.linear.router import build_lr_figure
from ..visualization.linear.prediction import explain_lr_prediction
from ..visualization.theme import attach_highlight

def visualize_lr(
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
    baseline="mean",
    display_space="original",
    metrics=None,
    dec=4,
    frame_duration=80,
    max_frames=60,
    frame_step=10,
    theme=None,
):
    """
    Generate an animated visualization for a linear regression model.

    This function is the primary public API of the library. It extracts the training
    history from the provided scikit-learn estimator and creates an interactive
    Plotly animation that demonstrates the evolution of the model's parameters and
    predictions across training steps.

    Args:
        trained_estimator: A fitted scikit-learn estimator or Pipeline.
        X (np.ndarray): The feature matrix used for training.
        y (np.ndarray): The target vector.
        steps (int, optional): The desired number of animation frames. Defaults to 60.
        mode (str, optional): Method to extract history ("auto", "iterative", "final_interp"). Defaults to "auto".
        show_loss (bool, optional): Whether to display the loss curve alongside the main plot. Defaults to True.
        title (str, optional): The title of the plot. Defaults to None.
        smooth (str, optional): Smoothing method for the loss curve (e.g., "ema" or None). Defaults to "ema".
        smooth_beta (float, optional): Beta parameter for EMA smoothing. Defaults to 0.85.
        strict_loss (bool, optional): If True, throw errors if loss cannot be animated cleanly. Defaults to False.
        baseline (str, optional): Initial reference line for the loss curve ("mean" or "zeros"). Defaults to "mean".
        display_space (str, optional): The space in which to display the parameters ("original" or "scaled"). Defaults to "original".
        dec (int, optional): The number of decimal places to format the parameters. Defaults to 4.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure object.
    """
    hist = fit_history(
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
    )

    fig = build_lr_figure(
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


__all__ = ["visualize_lr", "fit_history", "build_lr_figure", "explain_lr_prediction"]
