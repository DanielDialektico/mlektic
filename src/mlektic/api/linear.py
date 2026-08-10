"""Public API for linear-regression visualization."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ..services.linear_history import fit_history
from ..visualization.linear.prediction import explain_lr_prediction
from ..visualization.linear.router import build_lr_figure
from ..visualization.theme import annotate_history_semantics, attach_highlight, configure_animation


def visualize_lr(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto",
    show_loss=True,
    title=None,
    show_history_context=True,
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

    This is the primary public API for tabular linear models. Because the
    estimator is already fitted, Mlektic does not claim to recover its original
    ``fit`` history. Incremental estimators are replayed over a clone;
    non-incremental estimators use a labeled synthetic interpolation. The
    figure subtitle, slider, and ``layout.meta`` expose the resolved source.

    Args:
        trained_estimator: A fitted scikit-learn estimator or Pipeline.
        X (np.ndarray): The feature matrix used for training.
        y (np.ndarray): The target vector.
        steps (int, optional): Semantic checkpoints constructed before display
            decimation. This is not necessarily the number of rendered frames.
        mode (str, optional): ``"auto"`` resolves replay for incremental
            estimators and interpolation otherwise. ``"iterative"`` requests
            replay, while ``"final_interp"`` requests synthetic interpolation.
        show_loss (bool, optional): Whether to display the loss curve alongside the main plot. Defaults to True.
        title (str, optional): The title of the plot. Defaults to None.
        show_history_context (bool, optional): Whether to add the provenance and
            N/K subtitle below the title. Defaults to True. Slider labels and
            ``layout.meta`` retain the same history context when False.
        smooth (str, optional): Display smoothing for loss (``"ema"`` or
            ``None``). Raw values remain available as ``loss_raw``.
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
        max_frames (int | None, optional): Maximum semantic checkpoints retained
            for display. Source coordinates remain in history metadata.
        frame_step (int | None, optional): Source-position stride used only when
            ``max_frames`` is ``None``.
        theme (str | None, optional): Registered visualization theme. The only
            phase-0 theme is the backward-compatible ``"classic"`` default.

    Returns:
        plotly.graph_objects.Figure: The animated Plotly figure object.
    """
    X_array = np.asarray(X)
    dimensions = 1 if X_array.ndim == 1 else int(X_array.shape[1])
    if not all(isinstance(value, bool) for value in (show_loss, strict_loss, show_history_context)):
        raise TypeError("show_loss, strict_loss, and show_history_context must be boolean values.")
    if not isinstance(dec, int) or isinstance(dec, bool) or dec < 0:
        raise ValueError("dec must be a non-negative integer.")
    if not isinstance(frame_duration, Real) or isinstance(frame_duration, bool) or frame_duration < 0:
        raise ValueError("frame_duration must be a non-negative real number.")
    if transition_duration is not None and (
        not isinstance(transition_duration, Real)
        or isinstance(transition_duration, bool)
        or transition_duration < 0
    ):
        raise ValueError("transition_duration must be a non-negative real number or None.")
    if animation_mode not in {"auto", "native", "hybrid"}:
        raise ValueError("animation_mode must be 'auto', 'native', or 'hybrid'.")
    if not isinstance(interpolation_frames, int) or interpolation_frames < 1:
        raise ValueError("interpolation_frames must be a positive integer.")
    if fps is not None and (not isinstance(fps, Real) or isinstance(fps, bool) or fps <= 0):
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
    annotate_history_semantics(fig, hist, show_title=show_history_context)
    return attach_highlight(fig, theme=theme)


__all__ = ["visualize_lr", "fit_history", "build_lr_figure", "explain_lr_prediction"]
