"""Public API for linear-regression visualization."""

from __future__ import annotations

import numpy as np

from ..services.linear_history import fit_history
from ..visualization.linear.prediction import explain_lr_prediction
from ..visualization.linear.router import build_lr_figure
from ..visualization.theme import attach_highlight, configure_animation


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
    transition_duration=None,
    animation_mode="auto",
    fps=None,
    interpolation_frames=3,
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
        display_space (str, optional): Parameter display space ("original" or "scaled"). Defaults to "original".
        metrics (Sequence[str] | dict, optional): Built-in metric names or custom metric callables.
        dec (int, optional): The number of decimal places to format the parameters. Defaults to 4.
        frame_duration (int, optional): Native-frame duration in milliseconds.
            In hybrid mode it is divided by ``interpolation_frames`` when
            ``fps`` is omitted. Defaults to 80.
        transition_duration (int | None, optional): Interpolation duration for 2D
            traces. ``None`` derives a smooth value from ``frame_duration``; ``0``
            disables transitions.
        animation_mode (str, optional): ``"auto"`` enables hybrid trace-only
            animation for one-dimensional regression, ``"native"`` preserves
            semantic frames and dynamic LaTeX layout substitutions, and
            ``"hybrid"`` explicitly requests synchronized trace subframes.
        fps (int | None, optional): Visual frames per second in hybrid mode. If
            omitted, ``frame_duration`` is divided across the visual subframes.
            Values from 30 to 45 are recommended for Jupyter and Colab.
        interpolation_frames (int, optional): Visual intervals inserted between
            consecutive training checkpoints in hybrid mode. Defaults to 3.
        max_frames (int | None, optional): Maximum rendered frames after temporal decimation. Defaults to 60.
        frame_step (int | None, optional): Step used when ``max_frames`` is None. Defaults to 10.
        theme (str | None, optional): Visualization theme name. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure object.
    """
    X_array = np.asarray(X)
    dimensions = 1 if X_array.ndim == 1 else int(X_array.shape[1])
    if animation_mode not in {"auto", "native", "hybrid"}:
        raise ValueError("animation_mode must be 'auto', 'native', or 'hybrid'.")
    if not isinstance(interpolation_frames, int) or interpolation_frames < 1:
        raise ValueError("interpolation_frames must be a positive integer.")
    if fps is not None and (not isinstance(fps, (int, float)) or fps <= 0):
        raise ValueError("fps must be a positive number or None.")

    resolved_animation_mode = animation_mode
    if animation_mode == "auto":
        resolved_animation_mode = "hybrid" if dimensions == 1 else "native"
    if resolved_animation_mode == "hybrid" and dimensions != 1:
        raise ValueError("Hybrid animation is currently available for one-dimensional regression only.")

    visual_frame_duration = frame_duration
    if resolved_animation_mode == "hybrid":
        visual_frame_duration = (
            max(1, round(1000 / fps))
            if fps is not None
            else max(1, round(frame_duration / interpolation_frames))
        )

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
        frame_duration=visual_frame_duration,
        animation_mode=resolved_animation_mode,
        interpolation_frames=interpolation_frames,
        theme=theme,
    )

    configure_animation(fig, visual_frame_duration, transition_duration)
    return attach_highlight(fig, theme=theme)


__all__ = ["visualize_lr", "fit_history", "build_lr_figure", "explain_lr_prediction"]
