"""Estimator-backed mathematical contracts for tabular visualizations.

The functions in this module deliberately separate values that can be
verified from a fitted estimator from canonical teaching equations.  This
prevents Mlektic from presenting a convenient formula as an estimator's exact
private optimization rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from .adapters.sklearn import SklearnAdapter
from .utils.math import _sigmoid
from .utils.probability import multiclass_probabilities

_DETAIL_LEVELS = {"essential", "academic", "complete"}


def validate_math_options(
    X: Any,
    *,
    detail: str,
    show_objective: str | bool,
    show_regularization: str | bool,
    feature_names: Sequence[str] | None,
    sample_index: int | None,
) -> tuple[np.ndarray, list[str], int, bool, bool]:
    """Validate shared Phase-1 options and resolve their automatic values."""
    if not isinstance(detail, str) or detail not in _DETAIL_LEVELS:
        raise ValueError("detail must be 'essential', 'academic', or 'complete'.")
    for name, value in (
        ("show_objective", show_objective),
        ("show_regularization", show_regularization),
    ):
        if value != "auto" and not isinstance(value, bool):
            raise TypeError(f"{name} must be 'auto' or a boolean value.")

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    if X_array.ndim != 2 or X_array.shape[0] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix.")

    names = _resolve_input_feature_names(X, feature_names, X_array.shape[1])
    if sample_index is None:
        resolved_sample = 0
    elif not isinstance(sample_index, Integral) or isinstance(sample_index, bool):
        raise TypeError("sample_index must be an integer or None.")
    else:
        resolved_sample = int(sample_index)
    if not 0 <= resolved_sample < X_array.shape[0]:
        raise ValueError(f"sample_index must be between 0 and {X_array.shape[0] - 1}.")

    objective_visible = detail != "essential" if show_objective == "auto" else show_objective
    regularization_visible = detail == "complete" if show_regularization == "auto" else show_regularization
    return X_array, names, resolved_sample, bool(objective_visible), bool(regularization_visible)


def build_linear_math_contract(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    history: dict[str, Any],
    detail: str,
    show_objective: str | bool,
    show_regularization: str | bool,
    feature_names: Sequence[str] | None,
    sample_index: int | None,
    dec: int,
) -> dict[str, Any]:
    """Build an auditable mathematical description of a fitted linear model."""
    X_array, input_names, sample, objective_visible, regularization_visible = validate_math_options(
        X,
        detail=detail,
        show_objective=show_objective,
        show_regularization=show_regularization,
        feature_names=feature_names,
        sample_index=sample_index,
    )
    y_array = np.asarray(y, dtype=float).ravel()
    adapter = SklearnAdapter(estimator)
    preprocessing = _preprocessing_contract(adapter, X_array, input_names)
    coefficients_model, intercept_model = adapter.extract_linear_theta()
    if coefficients_model is None:
        raise ValueError("The fitted estimator does not expose a linear coefficient vector.")

    coefficients, intercept, values, names, equation_space = _display_linear_parameters(
        coefficients_model,
        intercept_model,
        X_array[sample],
        sample,
        preprocessing,
        history.get("display_space", "original"),
    )
    contributions = values * coefficients
    reconstructed = float(intercept + np.sum(contributions))
    model_prediction = float(np.asarray(adapter.predict(X_array[sample : sample + 1])).ravel()[0])
    predictions = np.asarray(adapter.predict(X_array), dtype=float).ravel()
    residuals = y_array - predictions
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    denominator = float(np.sum((y_array - np.mean(y_array)) ** 2))
    r2 = 0.0 if denominator <= 1e-12 else float(1.0 - np.sum(residuals**2) / denominator)
    regularization = _regularization_contract(adapter.final_estimator, family="linear")
    history_source = history.get("metadata", {}).get("source", history.get("history_source"))

    return {
        "schema_version": 1,
        "detail": detail,
        "family": "linear_regression",
        "estimator": adapter.final_estimator.__class__.__name__,
        "equation_space": equation_space,
        "requested_display_space": history.get("display_space", "original"),
        "feature_space": _public_preprocessing(preprocessing),
        "feature_names": names,
        "dimensions": {"samples": int(X_array.shape[0]), "features": int(coefficients.size)},
        "parameters": {
            "coefficients": coefficients.tolist(),
            "intercept": float(intercept),
            "intercept_separate": True,
        },
        "sample": {
            "index": sample,
            "values": values.tolist(),
            "contributions": contributions.tolist(),
            "contribution_sum": float(np.sum(contributions)),
            "reconstructed_prediction": reconstructed,
            "model_prediction": model_prediction,
            "matches_model": bool(np.isclose(reconstructed, model_prediction, rtol=1e-7, atol=1e-9)),
        },
        "objective": {
            "name": "mean_squared_error",
            "role": "empirical evaluation metric",
            "normalization": "1/n",
            "value": mse,
            "formula": "MSE(theta, theta_0) = (1/n) sum_i (y_i - y_hat_i)^2",
            "optimizer_relation": _linear_objective_relation(adapter.final_estimator),
            "visible": objective_visible,
        },
        "metrics": {"mse": mse, "mae": mae, "r2": r2},
        "regularization": {**regularization, "visible": regularization_visible},
        "update_rule": _update_rule_contract(adapter.final_estimator, history_source),
        "history_source": history_source,
        "panel": _linear_panel(
            coefficients,
            intercept,
            values,
            contributions,
            reconstructed,
            names,
            equation_space,
            mse,
            mae,
            r2,
            regularization,
            objective_visible,
            regularization_visible,
            detail,
            dec,
        ),
    }


def build_logistic_math_contract(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    history: dict[str, Any],
    detail: str,
    threshold: float,
    class_focus: Any,
    show_objective: str | bool,
    show_regularization: str | bool,
    feature_names: Sequence[str] | None,
    sample_index: int | None,
    show_class_labels: bool,
    dec: int,
) -> dict[str, Any]:
    """Build an auditable mathematical description of a fitted logistic model."""
    if not isinstance(threshold, Real) or isinstance(threshold, bool) or not 0 < float(threshold) < 1:
        raise ValueError("threshold must be a real number strictly between 0 and 1.")
    threshold = float(threshold)
    X_array, input_names, sample, objective_visible, regularization_visible = validate_math_options(
        X,
        detail=detail,
        show_objective=show_objective,
        show_regularization=show_regularization,
        feature_names=feature_names,
        sample_index=sample_index,
    )
    y_array = np.asarray(y).ravel()
    adapter = SklearnAdapter(estimator)
    classes = adapter.classes if adapter.classes is not None else np.unique(y_array)
    classes = np.asarray(classes)
    if classes.size < 2:
        raise ValueError("Logistic visualization requires at least two fitted classes.")
    focus_index = _resolve_class_focus(class_focus, classes)
    preprocessing = _preprocessing_contract(adapter, X_array, input_names)
    theta = adapter.extract_logistic_theta()
    if theta is None:
        raise ValueError("The fitted estimator does not expose logistic coefficients and intercepts.")

    is_multiclass = classes.size > 2
    probability_link = history.get("probability_link", "sigmoid" if not is_multiclass else "softmax")
    probabilities = np.asarray(adapter.predict_proba(X_array, classes), dtype=float)
    sample_probabilities = probabilities[sample]
    model_class_index = int(np.argmax(sample_probabilities))
    if is_multiclass:
        decision_index = model_class_index
    else:
        decision_index = 1 if float(sample_probabilities[1]) >= threshold else 0

    parameter_view = _display_logistic_parameters(
        theta,
        X_array[sample],
        sample,
        preprocessing,
        history.get("display_space", "original"),
    )
    weights = parameter_view["weights"]
    intercepts = parameter_view["intercepts"]
    values = parameter_view["values"]
    names = parameter_view["feature_names"]
    equation_space = parameter_view["equation_space"]
    if is_multiclass:
        scores = values @ weights + intercepts
        reconstructed_probabilities = multiclass_probabilities(
            np.asarray(scores, dtype=float).reshape(1, -1), probability_link
        )[0]
        contributions = values[:, None] * weights
    else:
        score = float(values @ weights + intercepts)
        scores = np.asarray([score])
        positive_probability = float(_sigmoid(score))
        reconstructed_probabilities = np.asarray([1.0 - positive_probability, positive_probability])
        contributions = values * weights

    clipped = np.clip(probabilities, 1e-15, 1.0)
    class_indices = np.array([int(np.flatnonzero(classes == label)[0]) for label in y_array])
    log_loss = float(-np.mean(np.log(clipped[np.arange(y_array.size), class_indices])))
    regularization = _regularization_contract(adapter.final_estimator, family="logistic")
    history_source = history.get("metadata", {}).get("source", history.get("history_source"))

    return {
        "schema_version": 1,
        "detail": detail,
        "family": "multiclass_logistic_regression" if is_multiclass else "binary_logistic_regression",
        "estimator": adapter.final_estimator.__class__.__name__,
        "equation_space": equation_space,
        "requested_display_space": history.get("display_space", "original"),
        "feature_space": _public_preprocessing(preprocessing),
        "feature_names": names,
        "dimensions": {
            "samples": int(X_array.shape[0]),
            "features": int(values.size),
            "classes": int(classes.size),
        },
        "classes": [_python_scalar(value) for value in classes],
        "class_order_visible": bool(show_class_labels),
        "class_focus_index": focus_index,
        "class_focus_label": None if focus_index is None else _python_scalar(classes[focus_index]),
        "probability_link": probability_link,
        "parameters": {
            "weights": np.asarray(weights, dtype=float).tolist(),
            "intercepts": np.atleast_1d(intercepts).astype(float).tolist(),
            "intercept_separate": True,
        },
        "sample": {
            "index": sample,
            "values": values.tolist(),
            "contributions": np.asarray(contributions, dtype=float).tolist(),
            "scores": np.asarray(scores, dtype=float).tolist(),
            "reconstructed_probabilities": reconstructed_probabilities.tolist(),
            "model_probabilities": sample_probabilities.tolist(),
            "matches_model": bool(
                np.allclose(reconstructed_probabilities, sample_probabilities, rtol=1e-7, atol=1e-9)
            ),
            "winning_class_index": decision_index,
            "winning_class_label": _python_scalar(classes[decision_index]),
            "estimator_winning_class_index": model_class_index,
        },
        "decision": {
            "rule": "argmax" if is_multiclass else "threshold",
            "threshold": None if is_multiclass else threshold,
            "positive_class_index": None if is_multiclass else 1,
            "positive_class_label": None if is_multiclass else _python_scalar(classes[1]),
            "source": "model argmax" if is_multiclass else "user threshold applied to model probability",
        },
        "objective": {
            "name": "multiclass_cross_entropy" if is_multiclass else "binary_log_loss",
            "role": "empirical data term",
            "normalization": "1/n",
            "value": log_loss,
            "formula": (
                "LogLoss = -(1/n) sum_i [y'_i log(p_i) + (1-y'_i) log(1-p_i)]"
                if not is_multiclass
                else "CrossEntropy = -(1/n) sum_i log(p_{i,y_i})"
            ),
            "optimizer_relation": (
                "The empirical data term is estimator-verifiable; exact private regularization scaling "
                "is not introspected."
            ),
            "visible": objective_visible,
        },
        "regularization": {**regularization, "visible": regularization_visible},
        "update_rule": _update_rule_contract(adapter.final_estimator, history_source),
        "history_source": history_source,
        "panel": _logistic_panel(
            classes,
            weights,
            intercepts,
            values,
            contributions,
            scores,
            reconstructed_probabilities,
            decision_index,
            focus_index,
            threshold,
            probability_link,
            names,
            equation_space,
            log_loss,
            regularization,
            objective_visible,
            regularization_visible,
            show_class_labels,
            detail,
            dec,
        ),
    }


def attach_math_contract(fig: Any, contract: dict[str, Any], *, theme: str | None = None) -> Any:
    """Attach mathematical metadata and an optional fitted-model reference panel."""
    metadata = dict(fig.layout.meta or {}) if isinstance(fig.layout.meta, dict) else {}
    metadata["mlektic_math"] = {key: value for key, value in contract.items() if key != "panel"}
    fig.update_layout(meta=metadata)
    if contract["detail"] == "essential":
        return fig

    panel_lines = contract["panel"]
    annotations = list(fig.layout.annotations or ())
    current_height = int(fig.layout.height or 600)
    current_margin = fig.layout.margin.to_plotly_json()
    margin_top = int(current_margin.get("t", 150))
    margin_bottom = int(current_margin.get("b", 70))
    plot_height = max(240, current_height - margin_top - margin_bottom)
    minimum_margin_bottom = 440 if contract["detail"] == "complete" else 350
    first_line_offset = 160
    panel_gap = 28
    previous_height = 0.0
    center_offset = float(first_line_offset)
    for line_index, line in enumerate(panel_lines):
        if isinstance(line, dict):
            text = str(line["text"])
            visual_lines = max(1, int(line.get("visual_lines", 1)))
        else:
            text = str(line)
            visual_lines = 1
        # Underbraces and feature labels make mathematical rows taller than
        # ordinary text. Reserve enough vertical space before placing the next
        # annotation so multiline derivations never collide with metric cards.
        estimated_height = 34.0 + 32.0 * (visual_lines - 1)
        if line_index:
            center_offset += previous_height / 2.0 + panel_gap + estimated_height / 2.0
        annotations.append(
            {
                "x": 0.5,
                "y": -center_offset / plot_height,
                "xref": "paper",
                "yref": "paper",
                "text": text,
                "showarrow": False,
                "xanchor": "center",
                "yanchor": "middle",
                "align": "center",
                "font": {"size": 13, "color": "white"},
                "bgcolor": "rgba(28,30,35,0.97)",
                "bordercolor": "rgba(184,193,204,0.55)",
                "borderwidth": 1,
                "borderpad": 7,
            }
        )
        previous_height = estimated_height
    required_margin_bottom = int(center_offset + previous_height / 2.0 + 60.0)
    current_margin["b"] = max(minimum_margin_bottom, required_margin_bottom)
    fig.update_layout(
        annotations=annotations,
        height=plot_height + margin_top + current_margin["b"],
        margin=current_margin,
    )
    return fig


def apply_logistic_focus_and_threshold(
    fig: Any,
    *,
    dimensions: int,
    classes: int,
    class_focus_index: int | None,
    threshold: float,
    detail: str,
) -> Any:
    """Apply optional geometry emphasis without changing the classic default."""
    if class_focus_index is not None and classes > 2 and dimensions <= 2:
        first_class_trace = 0 if dimensions == 1 else 1
        class_trace_indices = set(range(first_class_trace, first_class_trace + classes))
        selected_trace = first_class_trace + class_focus_index
        for trace_index, trace in enumerate(fig.data):
            if trace_index in class_trace_indices:
                trace.visible = trace_index == selected_trace
        for frame in fig.frames or ():
            trace_targets = list(frame.traces or range(len(frame.data)))
            for local_index, target_index in enumerate(trace_targets):
                if target_index in class_trace_indices and local_index < len(frame.data):
                    frame.data[local_index].visible = target_index == selected_trace
        title = fig.layout.title.text or ""
        fig.update_layout(title_text=f"{title} · class focus c_{class_focus_index} (1/{classes})")

    if classes != 2:
        return fig
    if detail == "essential" and np.isclose(threshold, 0.5):
        return fig
    _replace_threshold_tick(fig, dimensions=dimensions, threshold=threshold)
    if dimensions == 1:
        x_range = list(fig.layout.xaxis.range or (0.0, 1.0))
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=float(x_range[0]),
            x1=float(x_range[1]),
            y0=threshold,
            y1=threshold,
            line={"color": "rgba(255,255,255,0.75)", "width": 1.5, "dash": "dash"},
        )
    elif dimensions == 2:
        for trace in fig.data:
            if getattr(trace, "type", "") == "surface":
                trace.contours.z.update(
                    show=True,
                    start=threshold,
                    end=threshold,
                    size=1,
                    color="white",
                    width=4,
                    usecolormap=False,
                    highlight=False,
                )
        for frame in fig.frames or ():
            for trace in frame.data:
                if getattr(trace, "type", "") == "surface":
                    trace.contours.z.update(
                        show=True,
                        start=threshold,
                        end=threshold,
                        size=1,
                        color="white",
                        width=4,
                        usecolormap=False,
                        highlight=False,
                    )
    return fig


def _replace_threshold_tick(fig: Any, *, dimensions: int, threshold: float) -> None:
    """Keep probability axes consistent with the configured binary threshold."""
    axis = fig.layout.yaxis if dimensions == 1 else fig.layout.scene.zaxis if dimensions == 2 else None
    if axis is None:
        return
    previous = list(axis.ticktext or ("0", "0.5", "1"))
    while len(previous) < 3:
        previous.append(str(len(previous)))
    middle = f"{threshold:g} threshold"
    axis.update(tickvals=[0.0, threshold, 1.0], ticktext=[previous[0], middle, previous[-1]])


def _resolve_input_feature_names(X: Any, feature_names: Sequence[str] | None, dimensions: int) -> list[str]:
    if feature_names is None and hasattr(X, "columns"):
        feature_names = [str(value) for value in X.columns]
    if feature_names is None:
        return [f"x_{index + 1}" for index in range(dimensions)]
    if isinstance(feature_names, str) or not isinstance(feature_names, Sequence):
        raise TypeError("feature_names must be a sequence of strings or None.")
    names = [str(value) for value in feature_names]
    if len(names) != dimensions:
        raise ValueError(f"feature_names must contain exactly {dimensions} names.")
    if any(not name.strip() for name in names):
        raise ValueError("feature_names must not contain empty names.")
    return names


def _preprocessing_contract(adapter: SklearnAdapter, X: np.ndarray, input_names: list[str]) -> dict[str, Any]:
    transformed = np.asarray(adapter.transform_X(X), dtype=float)
    if transformed.ndim == 1:
        transformed = transformed.reshape(-1, 1)
    transformed_names = _transformed_feature_names(adapter, input_names, transformed.shape[1])
    steps = []
    if adapter.is_pipeline:
        steps = [
            {"name": name, "type": step.__class__.__name__}
            for name, step in adapter.estimator.steps[:-1]
        ]
    affine, matrix, offset = _infer_affine_transform(adapter, X, transformed)
    return {
        "pipeline": adapter.is_pipeline,
        "steps": steps,
        "input_dimension": int(X.shape[1]),
        "transformed_dimension": int(transformed.shape[1]),
        "input_feature_names": input_names,
        "transformed_feature_names": transformed_names,
        "is_affine": affine,
        "affine_matrix": None if matrix is None else matrix.tolist(),
        "affine_offset": None if offset is None else offset.tolist(),
        "raw_space_coefficients_available": affine,
        "statement": (
            "The preprocessing map was verified as affine on the supplied feature space."
            if affine
            else "Raw-space coefficients are not claimed; mathematics is shown in transformed-feature space."
        ),
        "_transformed_matrix": transformed,
    }


def _public_preprocessing(preprocessing: dict[str, Any]) -> dict[str, Any]:
    """Remove calculation-only arrays from the serialized feature-space contract."""
    return {key: value for key, value in preprocessing.items() if not key.startswith("_")}


def _transformed_feature_names(adapter: SklearnAdapter, input_names: list[str], dimensions: int) -> list[str]:
    if not adapter.is_pipeline:
        return input_names
    preprocessing = adapter.estimator[:-1]
    if hasattr(preprocessing, "get_feature_names_out"):
        try:
            names = [str(value) for value in preprocessing.get_feature_names_out(input_names)]
            if len(names) == dimensions:
                return names
        except (TypeError, ValueError, AttributeError):
            pass
    return [f"u_{index + 1}" for index in range(dimensions)]


def _infer_affine_transform(
    adapter: SklearnAdapter,
    X: np.ndarray,
    transformed: np.ndarray,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    if not adapter.is_pipeline:
        dimensions = X.shape[1]
        return True, np.eye(dimensions), np.zeros(dimensions)
    try:
        zero = np.zeros((1, X.shape[1]), dtype=float)
        basis = np.eye(X.shape[1], dtype=float)
        offset = np.asarray(adapter.transform_X(zero), dtype=float).reshape(-1)
        basis_transformed = np.asarray(adapter.transform_X(basis), dtype=float)
        matrix = (basis_transformed - offset).T
        reconstructed = X @ matrix.T + offset
        affine = reconstructed.shape == transformed.shape and np.allclose(
            reconstructed, transformed, rtol=1e-8, atol=1e-10
        )
        return bool(affine), matrix if affine else None, offset if affine else None
    except (TypeError, ValueError, AttributeError):
        return False, None, None


def _display_linear_parameters(
    coefficients: np.ndarray,
    intercept: float,
    raw_sample: np.ndarray,
    sample_index: int,
    preprocessing: dict[str, Any],
    requested_space: str,
) -> tuple[np.ndarray, float, np.ndarray, list[str], str]:
    coefficients = np.asarray(coefficients, dtype=float).ravel()
    if requested_space == "original" and preprocessing["is_affine"]:
        matrix = np.asarray(preprocessing["affine_matrix"], dtype=float)
        offset = np.asarray(preprocessing["affine_offset"], dtype=float)
        raw_coefficients = matrix.T @ coefficients
        raw_intercept = float(intercept + offset @ coefficients)
        return (
            raw_coefficients,
            raw_intercept,
            np.asarray(raw_sample, dtype=float),
            list(preprocessing["input_feature_names"]),
            "original",
        )
    selected = np.asarray(preprocessing["_transformed_matrix"][sample_index], dtype=float).ravel()
    return (
        coefficients,
        float(intercept),
        selected,
        list(preprocessing["transformed_feature_names"]),
        "transformed",
    )


def _display_logistic_parameters(
    theta: dict[str, Any],
    raw_sample: np.ndarray,
    sample_index: int,
    preprocessing: dict[str, Any],
    requested_space: str,
) -> dict[str, Any]:
    transformed_rows = np.asarray(preprocessing["_transformed_matrix"], dtype=float)
    if requested_space == "original" and preprocessing["is_affine"]:
        matrix = np.asarray(preprocessing["affine_matrix"], dtype=float)
        offset = np.asarray(preprocessing["affine_offset"], dtype=float)
        if theta["task"] == "binary":
            model_weights = np.asarray(theta["w"], dtype=float)
            weights = matrix.T @ model_weights
            intercepts: float | np.ndarray = float(theta["b"] + offset @ model_weights)
        else:
            model_weights = np.asarray(theta["W"], dtype=float)
            weights = matrix.T @ model_weights
            intercepts = np.asarray(theta["b"], dtype=float) + offset @ model_weights
        return {
            "weights": weights,
            "intercepts": intercepts,
            "values": np.asarray(raw_sample, dtype=float),
            "feature_names": list(preprocessing["input_feature_names"]),
            "equation_space": "original",
        }
    values = transformed_rows[sample_index]
    return {
        "weights": np.asarray(theta["w"] if theta["task"] == "binary" else theta["W"], dtype=float),
        "intercepts": float(theta["b"]) if theta["task"] == "binary" else np.asarray(theta["b"], dtype=float),
        "values": values,
        "feature_names": list(preprocessing["transformed_feature_names"]),
        "equation_space": "transformed",
    }


def _resolve_class_focus(class_focus: Any, classes: np.ndarray) -> int | None:
    if class_focus is None:
        return None
    matches = np.flatnonzero(classes == class_focus)
    if matches.size:
        return int(matches[0])
    if isinstance(class_focus, Integral) and not isinstance(class_focus, bool):
        index = int(class_focus)
        if 0 <= index < classes.size:
            return index
    raise ValueError(
        "class_focus must be a fitted class label, a zero-based class index, or None."
    )


def _regularization_contract(estimator: Any, *, family: str) -> dict[str, Any]:
    parameters = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    penalty = parameters.get("penalty")
    parameter_source = "penalty"
    if estimator.__class__.__name__ == "LogisticRegression" and penalty == "deprecated":
        ratio = float(parameters.get("l1_ratio", 0.0))
        if np.isinf(float(parameters.get("C", 1.0))):
            penalty = None
        elif np.isclose(ratio, 0.0):
            penalty = "l2"
        elif np.isclose(ratio, 1.0):
            penalty = "l1"
        else:
            penalty = "elasticnet"
        parameter_source = "l1_ratio (Scikit-learn 1.8+ effective convention)"
    if penalty in {None, "none"} or (family == "linear" and "penalty" not in parameters):
        return {
            "active": False,
            "type": "none",
            "parameter_source": parameter_source,
            "strength_parameter": None,
            "formula": "R(theta) = 0",
            "intercept_penalty": "not applicable",
            "claim": "No exposed penalty parameter is active.",
        }
    formulas = {
        "l2": "R(theta) = lambda ||theta||_2^2",
        "l1": "R(theta) = lambda ||theta||_1",
        "elasticnet": "R(theta) = lambda [rho ||theta||_1 + (1-rho) ||theta||_2^2]",
    }
    if "alpha" in parameters:
        strength = {"name": "alpha", "value": _python_scalar(parameters["alpha"])}
    elif "C" in parameters:
        strength = {"name": "C", "value": _python_scalar(parameters["C"]), "meaning": "inverse strength"}
    else:
        strength = {"name": "not introspected", "value": None}
    return {
        "active": True,
        "type": str(penalty),
        "parameter_source": parameter_source,
        "strength_parameter": strength,
        "l1_ratio": _python_scalar(parameters.get("l1_ratio")),
        "formula": formulas.get(str(penalty), "Estimator-specific penalty; formula not introspected."),
        "intercept_penalty": "not introspected",
        "claim": (
            "The penalty family and public strength parameter are estimator-backed; "
            "exact internal normalization is not claimed."
        ),
    }


def _linear_objective_relation(estimator: Any) -> str:
    if estimator.__class__.__name__ == "LinearRegression":
        return (
            "The estimator minimizes residual sum of squares. Displayed MSE differs by the positive constant 1/n "
            "and therefore has the same minimizer."
        )
    return (
        "Displayed MSE is estimator-verifiable. The estimator's exact private loss normalization, batching, "
        "averaging, and penalty combination are not introspected."
    )


def _update_rule_contract(estimator: Any, history_source: str | None) -> dict[str, Any]:
    incremental = hasattr(estimator, "partial_fit")
    return {
        "canonical": "theta^(t+1) = theta^t - eta_t grad J(theta^t)",
        "shown_as_exact_estimator_rule": False,
        "reason": (
            "The canonical gradient update is a teaching reference; estimator-private scheduling, batching, "
            "averaging, and penalty details are not fully introspected."
            if incremental
            else (
                "The fitted estimator does not expose a replayable step rule; the visible path is not "
                "optimizer history."
            )
        ),
        "history_source": history_source,
    }


def _linear_panel(
    coefficients: np.ndarray,
    intercept: float,
    values: np.ndarray,
    contributions: np.ndarray,
    prediction: float,
    names: list[str],
    equation_space: str,
    mse: float,
    mae: float,
    r2: float,
    regularization: dict[str, Any],
    objective_visible: bool,
    regularization_visible: bool,
    detail: str,
    dec: int,
) -> list[str | dict[str, Any]]:
    term_rows, visible_count = _contribution_rows(coefficients, values, names, dec)
    hidden = visible_count < coefficients.size
    state = (
        f"<b>Fitted-model derivation</b> · {equation_space} feature space · "
        f"x and theta in R^{coefficients.size}"
    )
    if coefficients.size <= 2:
        calculation = {
            "text": (
                rf"$\hat{{y}}=\theta_0+\mathbf{{x}}^\top\boldsymbol{{\theta}}"
                rf"={intercept:.{dec}f}+{term_rows[0]}={prediction:.{dec}f}$"
            ),
            "visual_lines": 1,
        }
    else:
        expansion = [
            r"\hat{y}&=\theta_0+\mathbf{x}^\top\boldsymbol{\theta}",
            rf"&={intercept:.{dec}f}+{term_rows[0]}",
        ]
        expansion.extend(rf"&\quad+{row}" for row in term_rows[1:])
        expansion.append(rf"&={prediction:.{dec}f}")
        calculation = {
            "text": r"$\begin{aligned}" + r"\\[4pt]".join(expansion) + r"\end{aligned}$",
            "visual_lines": len(expansion),
        }
    lines = [state, calculation]
    if hidden:
        lines.append(
            f"Showing {visible_count} of {coefficients.size} contributions, selected by "
            "absolute contribution magnitude; every value remains available in figure metadata."
        )
    if objective_visible:
        lines.append(
            rf"$\underbrace{{\mathrm{{MSE}}=\frac{{1}}{{n}}\sum_i(y_i-\hat{{y}}_i)^2={mse:.{dec}f}}}"
            rf"_{{\mathrm{{empirical\ evaluation}}}},\quad\mathrm{{MAE}}={mae:.{dec}f},\quad R^2={r2:.{dec}f}$"
        )
    if regularization_visible:
        lines.append(
            f"Regularization: {_regularization_panel_text(regularization)} · intercept penalty: "
            f"{regularization['intercept_penalty']}"
        )
    if detail == "complete":
        lines.append(
            "Canonical gradient updates are references unless the estimator-specific rule is fully introspected."
        )
    return lines


def _logistic_panel(
    classes: np.ndarray,
    weights: np.ndarray,
    intercepts: float | np.ndarray,
    values: np.ndarray,
    contributions: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    winner: int,
    class_focus_index: int | None,
    threshold: float,
    probability_link: str,
    names: list[str],
    equation_space: str,
    loss: float,
    regularization: dict[str, Any],
    objective_visible: bool,
    regularization_visible: bool,
    show_class_labels: bool,
    detail: str,
    dec: int,
) -> list[str | dict[str, Any]]:
    is_multiclass = classes.size > 2
    visible_order = (
        ", ".join(f"c_{index}={value}" for index, value in enumerate(classes))
        if show_class_labels
        else ", ".join(f"c_{index}" for index in range(classes.size))
    )
    state = (
        f"<b>Fitted-model derivation</b> · {equation_space} feature space · fitted order [{visible_order}] · "
        f"link: {probability_link}"
    )
    if is_multiclass:
        focus = winner if class_focus_index is None else class_focus_index
        term_rows, visible_count = _contribution_rows(weights[:, focus], values, names, dec)
        dimensions = int(weights.shape[0])
        hidden = visible_count < dimensions
        score = float(np.asarray(scores)[focus])
        intercept = float(np.asarray(intercepts)[focus])
        definition = rf"z_{{{focus}}}&=b_{{{focus}}}+\mathbf{{x}}^\top\mathbf{{w}}_{{{focus}}}"
    else:
        term_rows, visible_count = _contribution_rows(np.asarray(weights), values, names, dec)
        dimensions = int(np.asarray(weights).size)
        hidden = visible_count < dimensions
        score = float(np.asarray(scores).ravel()[0])
        intercept = float(intercepts)
        definition = r"z&=\theta_0+\mathbf{x}^\top\boldsymbol{\theta}"
    if dimensions <= 2 and is_multiclass:
        calculation = {
            "text": (
                rf"$z_{{{focus}}}=b_{{{focus}}}+\mathbf{{x}}^\top\mathbf{{w}}_{{{focus}}}"
                rf"={intercept:.{dec}f}+{term_rows[0]}={score:.{dec}f},\quad"
                rf"p_{{{focus}}}={float(probabilities[focus]):.{dec}f},\quad\hat{{y}}=c_{{{winner}}}$"
            ),
            "visual_lines": 1,
        }
    elif dimensions <= 2:
        calculation = {
            "text": (
                rf"$z={intercept:.{dec}f}+{term_rows[0]}={score:.{dec}f},\;"
                rf"p(Y=c_1\mid\mathbf{{x}})=\sigma(z)={float(probabilities[1]):.{dec}f},\;"
                rf"\tau={threshold:.{dec}f}\Rightarrow\hat{{y}}=c_{{{winner}}}$"
            ),
            "visual_lines": 1,
        }
    else:
        expansion = [definition, rf"&={intercept:.{dec}f}+{term_rows[0]}"]
        expansion.extend(rf"&\quad+{row}" for row in term_rows[1:])
        expansion.append(rf"&={score:.{dec}f}")
        if is_multiclass:
            expansion.append(
                rf"p_{{{focus}}}&={float(probabilities[focus]):.{dec}f},\qquad\hat{{y}}=c_{{{winner}}}"
            )
        else:
            expansion.extend(
                [
                    rf"p(Y=c_1\mid\mathbf{{x}})&=\sigma(z)={float(probabilities[1]):.{dec}f}",
                    rf"\tau&={threshold:.{dec}f}\Rightarrow\hat{{y}}=c_{{{winner}}}",
                ]
            )
        calculation = {
            "text": r"$\begin{aligned}" + r"\\[4pt]".join(expansion) + r"\end{aligned}$",
            "visual_lines": len(expansion),
        }
    lines = [state, calculation]
    if hidden:
        total = int(weights.shape[0] if is_multiclass else np.asarray(weights).size)
        lines.append(
            f"Showing {visible_count} of {total} contributions, selected by absolute contribution magnitude; "
            "every value remains available in figure metadata."
        )
    if objective_visible:
        loss_name = "Cross-entropy" if is_multiclass else "Log-loss"
        lines.append(
            rf"$\underbrace{{\mathrm{{{loss_name}}}= -\frac{{1}}{{n}}\sum_i \log p_{{i,y_i}}={loss:.{dec}f}}}"
            r"_{\mathrm{empirical\ data\ term}}\quad"
            r"\text{private penalty scaling is not claimed}$"
        )
    if regularization_visible:
        lines.append(
            f"Regularization: {_regularization_panel_text(regularization)} · intercept penalty: "
            f"{regularization['intercept_penalty']}"
        )
    if detail == "complete":
        lines.append(
            "Canonical gradient updates are references unless the estimator-specific rule is fully introspected."
        )
    return lines


def _contribution_rows(
    coefficients: np.ndarray,
    values: np.ndarray,
    names: list[str],
    dec: int,
    limit: int = 9,
    terms_per_row: int = 3,
) -> tuple[list[str], int]:
    """Format bounded, width-safe rows of coefficient-value products."""
    coefficients = np.asarray(coefficients, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    count = coefficients.size
    if count <= limit:
        indices = list(range(count))
    else:
        order = np.argsort(np.abs(coefficients * values))[::-1]
        indices = sorted(order[:limit].tolist())
    terms = [
        rf"\underbrace{{({coefficients[index]:.{dec}f})({values[index]:.{dec}f})}}"
        rf"_{{\mathrm{{{_latex_label(names[index])}}}}}"
        for index in indices
    ]
    rows = [
        "+".join(terms[start : start + terms_per_row])
        for start in range(0, len(terms), terms_per_row)
    ]
    if count > len(indices):
        rows[-1] += r"+\cdots"
    return rows, len(indices)


def _regularization_panel_text(regularization: dict[str, Any]) -> str:
    """Format only estimator-backed public regularization settings."""
    strength = regularization.get("strength_parameter")
    if not strength:
        return str(regularization["formula"])
    detail = f"{strength['name']}={strength['value']}"
    if strength.get("meaning"):
        detail += f" ({strength['meaning']})"
    if regularization.get("l1_ratio") is not None:
        detail += f", l1_ratio={regularization['l1_ratio']}"
    return f"{regularization['formula']} · {detail}"


def _latex_label(value: str) -> str:
    """Escape the small subset of characters used in feature-name subscripts."""
    return str(value).replace("\\", "").replace("_", r"\_").replace("%", r"\%")


def _python_scalar(value: Any) -> Any:
    if value is None:
        return None
    return value.item() if hasattr(value, "item") else value


__all__ = [
    "apply_logistic_focus_and_threshold",
    "attach_math_contract",
    "build_linear_math_contract",
    "build_logistic_math_contract",
    "validate_math_options",
]
