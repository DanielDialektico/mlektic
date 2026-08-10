"""Routing logic for logistic-regression figure selection."""

from __future__ import annotations

import numpy as np

from ..._internal.common import _first_not_none
from .binary_1d import build_binary_simple_logistic_figure
from .binary_2d import build_binary_plane_logistic_figure
from .binary_nd import build_binary_multivar_logistic_figure
from .multiclass_1d import build_multiclass_1d_logistic_figure
from .multiclass_2d import build_multiclass_2d_logistic_figure
from .multiclass_nd import build_multiclass_multivar_logistic_figure


def _attach_class_display_metadata(fig, classes, show_class_labels):
    """Preserve fitted class semantics independently of visible labels."""
    metadata = dict(fig.layout.meta or {}) if isinstance(fig.layout.meta, dict) else {}
    metadata["mlektic_classes"] = {
        "classes": [value.item() if hasattr(value, "item") else value for value in classes],
        "show_class_labels": show_class_labels,
    }
    fig.update_layout(meta=metadata)
    return fig


def build_logistic_figure(
    X,
    y,
    w_hist=None,
    b_hist=None,
    *,
    history=None,
    p_line_hist=None,
    x1_grid=None,  # binary d==1
    p_plane_hist=None,
    X1g=None,
    X2g=None,  # binary d==2
    p_curves_hist=None,  # multiclass d==1
    p_surfaces_hist=None,  # multiclass d==2
    loss_hist=None,
    classes=None,
    show_loss=False,
    show_class_labels=False,
    history_kind="iterative",
    title=None,
    strict_loss=False,
    dec=4,
    frame_duration=80,
    max_theta_cols=8,
    probability_link="softmax",
    theme=None,
):
    """Build logistic figure based on data dimensionality."""
    if not isinstance(show_class_labels, bool):
        raise TypeError("show_class_labels must be a boolean value.")
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    d = int(X.shape[1])

    if history is not None:
        if not isinstance(history, dict):
            raise ValueError("history must be a dict returned by fit_history_logistic().")

        history_kind = history.get("history_kind", history_kind)
        classes = _first_not_none(history.get("classes", None), classes)
        loss_hist = _first_not_none(
            history.get("loss_hist", None),
            history.get("losses", None),
            history.get("loss", None),
            loss_hist,
        )

        grid = history.get("grid", {}) or {}

        metrics_hist = history.get("metrics_hist", None)
        probability_link = history.get("probability_link", probability_link)

        w_hist = _first_not_none(history.get("w_hist", None), w_hist)
        b_hist = _first_not_none(history.get("b_hist", None), b_hist)

        p_line_hist = _first_not_none(history.get("p_line_hist", None), p_line_hist)
        p_plane_hist = _first_not_none(history.get("p_plane_hist", None), p_plane_hist)
        p_curves_hist = _first_not_none(history.get("p_curves_hist", None), p_curves_hist)
        p_surfaces_hist = _first_not_none(history.get("p_surfaces_hist", None), p_surfaces_hist)

        x1_grid = _first_not_none(grid.get("x1_grid", None), x1_grid)
        X1g = _first_not_none(grid.get("X1g", None), X1g)
        X2g = _first_not_none(grid.get("X2g", None), X2g)

    classes = np.unique(y) if classes is None else np.asarray(classes)
    K = len(classes)
    is_multiclass = K > 2

    if history is None:
        metrics_hist = None

    if not is_multiclass:
        y_binary = (y == classes[1]).astype(float)
        if d == 1:
            x1 = X[:, 0]
            if title is None:
                title = "Binary Logistic Regression (1 variable)"
            fig = build_binary_simple_logistic_figure(
                x1,
                y_binary,
                w_hist=w_hist,
                b_hist=b_hist,
                p_line_hist=p_line_hist,
                x1_grid=x1_grid,
                loss_hist=loss_hist,
                metrics_hist=metrics_hist,
                show_loss=show_loss,
                classes=classes,
                show_class_labels=show_class_labels,
                history_kind=history_kind,
                title=title,
                strict_loss=strict_loss,
                dec=dec,
                frame_duration=frame_duration,
                theme=theme,
            )
            return _attach_class_display_metadata(fig, classes, show_class_labels)

        if d == 2:
            x1 = X[:, 0]
            x2 = X[:, 1]
            if title is None:
                title = "Binary Logistic Regression (2 variables)"
            fig = build_binary_plane_logistic_figure(
                x1,
                x2,
                y_binary,
                w_hist=w_hist,
                b_hist=b_hist,
                p_plane_hist=p_plane_hist,
                X1g=X1g,
                X2g=X2g,
                loss_hist=loss_hist,
                metrics_hist=metrics_hist,
                show_loss=show_loss,
                classes=classes,
                show_class_labels=show_class_labels,
                history_kind=history_kind,
                title=title,
                strict_loss=strict_loss,
                dec=dec,
                frame_duration=frame_duration,
                theme=theme,
            )
            return _attach_class_display_metadata(fig, classes, show_class_labels)

        if title is None:
            title = f"Binary Logistic Regression ({d} variables)"
        fig = build_binary_multivar_logistic_figure(
            X,
            y,
            w_hist,
            b_hist,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            classes=classes,
            show_class_labels=show_class_labels,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=dec,
            frame_duration=frame_duration,
            theme=theme,
        )
        return _attach_class_display_metadata(fig, classes, show_class_labels)

    # multiclass
    if d == 1:
        if title is None:
            title = f"Multiclass Logistic Regression (K={K}, d=1)"
        fig = build_multiclass_1d_logistic_figure(
            X[:, 0],
            y,
            w_hist,
            b_hist,
            p_curves_hist=p_curves_hist,
            x1_grid=x1_grid,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            classes=classes,
            show_class_labels=show_class_labels,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=min(dec, 4),
            frame_duration=frame_duration,
            max_theta_cols=max_theta_cols,
            probability_link=probability_link,
            theme=theme,
        )
        return _attach_class_display_metadata(fig, classes, show_class_labels)

    if d == 2:
        if title is None:
            title = f"Multiclass Logistic Regression (K={K}, d=2)"
        fig = build_multiclass_2d_logistic_figure(
            X[:, 0],
            X[:, 1],
            y,
            w_hist,
            b_hist,
            p_surfaces_hist=p_surfaces_hist,
            X1g=X1g,
            X2g=X2g,
            loss_hist=loss_hist,
            metrics_hist=metrics_hist,
            show_loss=show_loss,
            classes=classes,
            show_class_labels=show_class_labels,
            history_kind=history_kind,
            title=title,
            strict_loss=strict_loss,
            dec=min(dec, 4),
            frame_duration=frame_duration,
            max_theta_cols=max_theta_cols,
            probability_link=probability_link,
            theme=theme,
        )
        return _attach_class_display_metadata(fig, classes, show_class_labels)

    if title is None:
        title = f"Multiclass Logistic Regression (K={K}, d={d})"
    fig = build_multiclass_multivar_logistic_figure(
        X,
        y,
        w_hist,
        b_hist,
        loss_hist=loss_hist,
        metrics_hist=metrics_hist,
        show_loss=show_loss,
        classes=classes,
        show_class_labels=show_class_labels,
        history_kind=history_kind,
        title=title,
        strict_loss=strict_loss,
        dec=min(dec, 4),
        frame_duration=frame_duration,
        max_theta_cols=max_theta_cols,
        probability_link=probability_link,
        theme=theme,
    )
    return _attach_class_display_metadata(fig, classes, show_class_labels)


__all__ = ["build_logistic_figure"]
