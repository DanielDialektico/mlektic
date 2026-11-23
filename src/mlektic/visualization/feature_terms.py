r"""LaTeX for models that are linear in transformed features ``u = φ(x)``.

A pipeline such as ``PolynomialFeatures(2) -> LogisticRegression`` is linear in
the transformed coordinates :math:`\mathbf u=\phi(\mathbf x)\in\mathbb R^{D_\phi}`,
not in the raw inputs. Builders keep plotting in the raw input space (curves and
surfaces come from the estimator itself) and write their equations with these
helpers, using the canonical feature names ``x_1``, ``x_1^2``, ``x_1 x_2`` that
``get_feature_names_out`` produces. Long sums keep their first terms and the last
one around ``\cdots``, so an equation stays one line for any :math:`D_\phi`.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

_CANONICAL = re.compile(r"([xu])_(\d+)(?:\^(\d+))?")


def term_latex(name: str) -> str:
    """Canonical feature name (``x_1``, ``x_1^2``, ``x_1 x_2``) as LaTeX."""
    value = str(name).replace("\\", "").strip()
    parts = [_CANONICAL.fullmatch(part) for part in value.split()]
    if parts and all(parts):
        latex = []
        for match in parts:
            symbol, index, power = match.groups()
            latex.append(rf"{symbol}_{{{index}}}" + (rf"^{{{power}}}" if power else ""))
        return "".join(latex)
    match = re.fullmatch(r"(.+?)\^(\d+)", value)
    if match:
        base, power = match.groups()
        escaped = base.replace("_", r"\_").replace("%", r"\%").replace(" ", r"\,")
        return rf"\mathrm{{{escaped}}}^{{{power}}}"
    escaped = value.replace("_", r"\_").replace("%", r"\%").replace(" ", r"\,")
    return rf"\mathrm{{{escaped}}}"


def visible_indices(count: int, maximum: int) -> tuple[list[int], bool]:
    """Indices kept in a truncated sum: the first ``maximum - 1`` and the last."""
    if count <= maximum:
        return list(range(count)), False
    return list(range(maximum - 1)) + [count - 1], True


def _join(terms: list[str], omitted: bool) -> str:
    if omitted:
        return "+".join(terms[:-1]) + r"+\cdots+" + terms[-1]
    return "+".join(terms)


def symbolic_sum_latex(
    names: Sequence[str],
    coefficient: Callable[[int], str],
    intercept: str,
    *,
    maximum: int = 4,
) -> str:
    r"""``θ0 + θ1 x_1 + θ2 x_1^2`` with ``coefficient(j)`` for the 1-based index ``j``."""
    indices, omitted = visible_indices(len(names), maximum)
    terms = [coefficient(index + 1) + term_latex(names[index]) for index in indices]
    return intercept + "+" + _join(terms, omitted)


def numeric_sum_latex(
    weights: Sequence[float],
    bias: float,
    names: Sequence[str],
    dec: int,
    *,
    maximum: int = 4,
    signed: bool = False,
) -> str:
    """The fitted affine score ``(b)+(w1)x_1+(w2)x_1^2`` with ``dec`` decimals."""
    fmt = f"{{:+.{dec}f}}" if signed else f"{{:.{dec}f}}"
    indices, omitted = visible_indices(len(names), maximum)
    terms = [rf"\left({fmt.format(float(weights[index]))}\right){term_latex(names[index])}" for index in indices]
    return rf"\left({fmt.format(float(bias))}\right)+" + _join(terms, omitted)


def feature_map_latex(names: Sequence[str], *, maximum: int = 4, raw: str = r"\mathbf x") -> str:
    r"""``\mathbf u=\phi(\mathbf x)=(x_1,x_1^2,\ldots)\in\mathbb R^{D}``."""
    indices, omitted = visible_indices(len(names), maximum)
    items = [term_latex(names[index]) for index in indices]
    if omitted:
        items = items[:-1] + [r"\ldots"] + items[-1:]
    return rf"\mathbf u=\phi({raw})=(" + ",".join(items) + rf")\in\mathbb R^{{{len(names)}}}"


_SUBSCRIPTED_X = re.compile(r"x_(?:\{(\d+)\}|(\d)(?![\d}])|\{([a-z])\}|([a-z]))")


def _inside_command(text: str, position: int) -> bool:
    r"""Whether the ``x`` at ``position`` ends a command name such as ``\max``."""
    start = position
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    return start > 0 and text[start - 1] == "\\"


def rename_features_latex(text: str, names: Sequence[str]) -> str:
    r"""Rewrite ``x_j`` of a coefficient view built on ``u = φ(x)``.

    Numbered coordinates ``x_{j}`` become the canonical term of feature ``j``
    (``x_1^{2}``, ``x_1x_2``), a generic ``x_j`` becomes ``u_j``, and the vector
    ``\mathbf{x}`` becomes ``\mathbf{u}``.
    """
    if "$" not in text:
        return text

    def replace(match: re.Match) -> str:
        if _inside_command(match.string, match.start()):
            return match.group(0)
        number = match.group(1) or match.group(2)
        if number is None:
            return rf"u_{{{match.group(3) or match.group(4)}}}"
        index = int(number)
        if 1 <= index <= len(names):
            term = term_latex(names[index - 1])
            return term if term.startswith(("x_", "u_")) else "{" + term + "}"
        return match.group(0)

    text = _SUBSCRIPTED_X.sub(replace, text)
    return text.replace(r"\mathbf{x}", r"\mathbf{u}").replace(r"\mathbf x", r"\mathbf u")


def rename_features_in_figure(fig, names: Sequence[str]) -> None:
    """Apply :func:`rename_features_latex` to every annotation and text trace."""

    def fix_annotations(annotations) -> None:
        for annotation in annotations or ():
            if annotation.text:
                annotation.text = rename_features_latex(str(annotation.text), names)

    def fix_traces(traces) -> None:
        for trace in traces or ():
            text = getattr(trace, "text", None)
            if isinstance(text, str):
                trace.text = rename_features_latex(text, names)
            elif text is not None and not hasattr(text, "dtype") and any("$" in str(v) for v in text):
                trace.text = [rename_features_latex(str(v), names) for v in text]

    fix_annotations(fig.layout.annotations)
    fix_traces(fig.data)
    for frame in fig.frames or ():
        if frame.layout is not None:
            fix_annotations(frame.layout.annotations)
        fix_traces(frame.data)


def rename_features_in_staged_figure(fig, names: Sequence[str]) -> None:
    """Rename features in a figure and in the annotation sets of its stage buttons."""
    rename_features_in_figure(fig, names)
    for menu in fig.layout.updatemenus or ():
        for button in menu.buttons or ():
            arguments = list(button.args or ())
            for argument in arguments:
                if not isinstance(argument, dict):
                    continue
                for key, value in argument.items():
                    if (key == "annotations" or key.endswith(".annotations")) and isinstance(value, (list, tuple)):
                        for annotation in value:
                            if isinstance(annotation, dict) and isinstance(annotation.get("text"), str):
                                annotation["text"] = rename_features_latex(annotation["text"], names)
            button.args = arguments


def transformed_feature_names(estimator, input_dimension: int) -> list[str]:
    """Canonical names of ``u = φ(x)``: ``x_1``, ``x_1^2``, ``x_1 x_2``, or ``u_j``."""
    symbolic = [f"x_{index + 1}" for index in range(input_dimension)]
    preprocessing = estimator[:-1]
    try:
        names = [str(name) for name in preprocessing.get_feature_names_out(symbolic)]
    except (AttributeError, TypeError, ValueError):
        names = []
    probe = preprocessing.transform(np.zeros((1, input_dimension)))
    width = int(np.asarray(probe).reshape(1, -1).shape[1])
    if len(names) != width or names == symbolic:
        names = [f"u_{index + 1}" for index in range(width)]
    return names


def value_rows_latex(names: Sequence[str], values: Sequence[float], dec: int, *, maximum: int = 4) -> list[str]:
    r"""``x_1^{2} &= 6.2500`` rows for an ``aligned`` block, with ``\vdots`` when cut."""
    indices, omitted = visible_indices(len(names), maximum)
    rows = [rf"{term_latex(names[index])} &= {float(values[index]):.{dec}f}" for index in indices]
    if omitted:
        rows = rows[:-1] + [r"&\;\vdots"] + rows[-1:]
    return rows


def transformed_input_latex(
    x_values, names: Sequence[str], u_values, dec: int, *, maximum: int = 4, fmt: Callable[[float], str] | None = None
) -> str:
    r"""Input stage of a prediction in ``u = φ(x)``: three lines for any width.

    ``\mathbf x`` with its values, ``\mathbf u=\phi(\mathbf x)`` with its values,
    and the transformed terms in the same order, long vectors cut around
    ``\ldots``.
    """

    def vector(items: list[str]) -> str:
        indices, omitted = visible_indices(len(items), maximum)
        shown = [items[index] for index in indices]
        if omitted:
            shown = shown[:-1] + [r"\ldots"] + shown[-1:]
        return "(" + ",\\,".join(shown) + ")"

    number = fmt or (lambda value: f"{float(value):.{dec}f}")
    x_items = [number(value) for value in np.ravel(x_values)]
    u_items = [number(value) for value in np.ravel(u_values)]
    terms = [term_latex(name) for name in names]
    return (
        r"$\begin{aligned}"
        + rf"\mathbf x &= {vector(x_items)}\\"
        + rf"\mathbf u=\phi(\mathbf x) &= {vector(u_items)}\\"
        + rf"&\;\;\,{vector(terms)}"
        + r"\end{aligned}$"
    )


def substitution_rows_latex(
    weights: Sequence[float],
    bias: float,
    values: Sequence[float],
    dec: int,
    *,
    maximum: int = 4,
    fmt: Callable[[float], str] | None = None,
) -> list[str]:
    r"""Rows ``&= (θ0) + (θ1)\cdot(u1)``, ``&\quad + (θj)\cdot(uj)`` of a substitution.

    A cut sum ends with one row ``+ \cdots + (θD)\cdot(uD)``, so the block keeps
    at most ``maximum`` rows whatever the number of features.
    """
    number = fmt or (lambda value: f"{float(value):.{dec}f}")
    indices, omitted = visible_indices(len(values), maximum)
    terms = [rf"({number(weights[index])}) \cdot ({number(values[index])})" for index in indices]
    rows = [rf"&= ({number(bias)}) + {terms[0]}"]
    middle = terms[1:-1] if omitted else terms[1:]
    rows += [rf"&\quad + {term}" for term in middle]
    if omitted:
        rows.append(rf"&\quad + \cdots + {terms[-1]}")
    return rows


def transformed_feature_space(estimator, X, math_contract) -> dict:
    """Builder keywords for a model whose equations live in ``u = φ(x)``.

    Returns ``{}`` when the mathematics is in the original feature space
    (no pipeline, or only affine scaling). Otherwise returns the canonical
    transformed names and ``u = φ(X)``. A non-affine map that keeps the input
    width and its names (for example a quantile transform) is renamed ``u_j``
    so its equations are never mistaken for raw ``x_j``.
    """
    from ..adapters.sklearn import SklearnAdapter

    if math_contract.get("equation_space") != "transformed":
        return {}
    names = [str(name) for name in math_contract.get("symbolic_feature_names", [])]
    raw_names = list(math_contract.get("feature_space", {}).get("input_symbolic_feature_names", []))
    if not names:
        return {}
    if names == raw_names:
        names = [f"u_{index + 1}" for index in range(len(names))]
    transformed = np.asarray(SklearnAdapter(estimator).transform_X(np.asarray(X, dtype=float)), dtype=float)
    if transformed.ndim == 1:
        transformed = transformed.reshape(-1, 1)
    return {"feature_names": names, "transformed_X": transformed}


__all__ = [
    "feature_map_latex",
    "rename_features_in_figure",
    "rename_features_in_staged_figure",
    "substitution_rows_latex",
    "transformed_feature_names",
    "transformed_input_latex",
    "transformed_feature_space",
    "value_rows_latex",
    "rename_features_latex",
    "numeric_sum_latex",
    "symbolic_sum_latex",
    "term_latex",
    "visible_indices",
]
