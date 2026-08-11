"""Public API for logistic-regression visualization."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ..mathematics import (
    apply_logistic_focus_and_threshold,
    attach_math_contract,
    build_logistic_math_contract,
)
from ..services.logistic_history import fit_history_logistic
from ..visualization.design import apply_visual_system, resolve_visual_spec
from ..visualization.logistic.prediction import explain_logistic_prediction
from ..visualization.logistic.router import build_logistic_figure
from ..visualization.theme import (
    annotate_history_semantics,
    annotate_loss_semantics,
    attach_highlight,
    configure_animation,
)


def visualize_logistic(
    trained_estimator,
    X,
    y,
    *,
    steps=60,
    mode="auto",
    show_loss=False,
    title=None,
    show_history_context=True,
    show_class_labels=False,
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
    detail="essential",
    threshold=0.5,
    class_focus=None,
    show_objective="auto",
    show_regularization="auto",
    feature_names=None,
    sample_index=None,
    theme=None,
    format="dashboard",
    density=None,
    size="default",
    width=None,
    height=None,
    responsive=False,
    reduced_motion=False,
):
    """Visualize a logistic-regression learning or interpolation history.

    Binary models use the sigmoid link. For multiclass models,
    ``multiclass_link="auto"`` compares decision scores with ``predict_proba``
    and selects either multinomial Softmax or normalized one-vs-rest sigmoids.

    An already fitted estimator does not expose its original fit trajectory.
    Incremental estimators are replayed over a clone and close with an
    explicitly labeled exact fitted endpoint; other estimators use a labeled
    synthetic interpolation. ``steps`` controls constructed semantic states,
    while ``max_frames`` and ``frame_step`` bound the displayed states and
    preserve their source coordinates and origins. ``transition_duration``
    controls visual interpolation between 2D frames; Plotly 3D traces retain
    redraw semantics.

    Args:
        trained_estimator: Fitted Scikit-Learn classifier or pipeline.
        X: Training feature matrix.
        y: Training class labels.
        steps: Number of captured states before temporal decimation.
        mode: ``"auto"``, ``"iterative"``, or ``"final_interp"``.
        show_loss: Whether to show empirical path evaluation. Replays and
            synthetic interpolations are labeled distinctly; neither is
            presented as an introspected private optimizer loss.
        title: Optional figure title.
        show_history_context: Whether to add the provenance and N/K subtitle
            below the title. Slider labels and ``layout.meta`` retain the same
            context when this is ``False``.
        show_class_labels: Whether to expose fitted semantic labels in axes and
            legends. The default ``False`` uses class indices while preserving
            the fitted labels in ``layout.meta``.
        smooth: ``"ema"`` or ``None`` for displayed replay loss. Synthetic
            interpolation displays raw empirical evaluation because its path
            is already smooth and its endpoint should remain exact. Raw loss
            remains in the history payload.
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
        detail: ``"essential"`` preserves the classic figure, ``"academic"``
            adds a fitted-model derivation, and ``"complete"`` adds the full
            objective, preprocessing, regularization, and optimizer caveats.
        threshold: Binary decision threshold, strictly between zero and one.
            Non-default thresholds are reflected in the academic decision and
            probability geometry. Multiclass decisions continue to use argmax.
        class_focus: Optional fitted label or zero-based class index. In 1D/2D
            multiclass figures only that probability curve or surface remains
            visible; the fitted class order remains in metadata.
        show_objective: ``"auto"`` or bool controlling the empirical log-loss
            or cross-entropy line in the mathematical panel.
        show_regularization: ``"auto"`` or bool controlling the estimator-backed
            regularization summary. Private normalization is not inferred.
        feature_names: Optional original feature-name sequence. DataFrame
            columns are used automatically.
        sample_index: Training observation used for the visible substitution;
            ``None`` selects index 0.
        theme: Optional visualization theme.
        format: ``"dashboard"``, ``"lesson"``, ``"compact"``, or the static
            final-state ``"report"`` composition.
        density: Optional alias for mathematical ``detail``.
        size: Named canvas preset: ``"default"``, ``"compact"``,
            ``"notebook"``, ``"wide"``, or ``"classroom"``.
        width: Optional explicit canvas width in pixels.
        height: Optional explicit canvas height in pixels.
        responsive: Scale the resolved composition with its container.
        reduced_motion: Show the exact final state without animation controls.

    Returns:
        An animated Plotly figure with mathematically matched definitions.
    """
    visual_spec = resolve_visual_spec(
        detail=detail,
        theme=theme,
        format=format,
        density=density,
        size=size,
        width=width,
        height=height,
        responsive=responsive,
        reduced_motion=reduced_motion,
    )
    detail = visual_spec.density
    if not all(
        isinstance(value, bool)
        for value in (show_loss, strict_loss, show_history_context, show_class_labels)
    ):
        raise TypeError(
            "show_loss, strict_loss, show_history_context, and show_class_labels "
            "must be boolean values."
        )
    if not isinstance(dec, int) or isinstance(dec, bool) or dec < 0:
        raise ValueError("dec must be a non-negative integer.")
    if not isinstance(max_theta_cols, int) or isinstance(max_theta_cols, bool) or max_theta_cols < 1:
        raise ValueError("max_theta_cols must be a positive integer.")
    if not isinstance(frame_duration, Real) or isinstance(frame_duration, bool) or frame_duration < 0:
        raise ValueError("frame_duration must be a non-negative real number.")
    if transition_duration is not None and (
        not isinstance(transition_duration, Real)
        or isinstance(transition_duration, bool)
        or transition_duration < 0
    ):
        raise ValueError("transition_duration must be a non-negative real number or None.")
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

    math_contract = build_logistic_math_contract(
        trained_estimator,
        X,
        y,
        history=hist,
        detail=detail,
        threshold=threshold,
        class_focus=class_focus,
        show_objective=show_objective,
        show_regularization=show_regularization,
        feature_names=feature_names,
        sample_index=sample_index,
        show_class_labels=show_class_labels,
        dec=dec,
    )

    fig = build_logistic_figure(
        X,
        y,
        history=hist,
        show_loss=show_loss,
        show_class_labels=show_class_labels,
        title=title,
        strict_loss=strict_loss,
        dec=dec,
        frame_duration=frame_duration,
        max_theta_cols=max_theta_cols,
        theme=theme,
    )

    X_array = np.asarray(X)
    dimensions = 1 if X_array.ndim == 1 else int(X_array.shape[1])
    apply_logistic_focus_and_threshold(
        fig,
        dimensions=dimensions,
        classes=math_contract["dimensions"]["classes"],
        class_focus_index=math_contract["class_focus_index"],
        threshold=float(threshold),
        detail=detail,
    )

    configure_animation(fig, frame_duration, transition_duration)
    annotate_history_semantics(fig, hist, show_title=show_history_context)
    annotate_loss_semantics(fig, hist)
    attach_math_contract(fig, math_contract, theme=theme)
    apply_visual_system(fig, visual_spec, family="logistic")
    return attach_highlight(fig, theme=theme)


__all__ = [
    "visualize_logistic",
    "fit_history_logistic",
    "build_logistic_figure",
    "explain_logistic_prediction",
]
