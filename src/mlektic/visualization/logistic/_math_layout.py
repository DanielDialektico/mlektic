"""Shared mathematical-layout tokens for logistic-regression figures."""

MULTICLASS_PROBABILITY_ROW_GAP = r"\\[10pt]"
MULTICLASS_PROBABILITY_FONT_SIZE = 13
MULTICLASS_ELLIPSIS_FONT_SIZE = 22
# Play/Pause sit higher than the default (1.14) so the full-width model
# formula below the subtitle never touches the buttons.
MULTICLASS_MENU_Y = 1.19


def compact_probability_fraction_latex(class_index: int, classes: int, probability_link: str) -> str:
    """Return an exact compact multiclass probability fraction.

    The expanded score remains visible separately.  This form is used only in
    constrained layouts where repeating every numeric score in the denominator
    would cross into an adjacent plot.
    """
    if probability_link == "ovr":
        return rf"\frac{{q_{{{class_index}}}}}{{\sum_{{r=0}}^{{K-1}}q_r}},\quad q_r=\sigma(z_r)"
    return rf"\frac{{e^{{z_{{{class_index}}}}}}}{{\sum_{{r=0}}^{{K-1}}e^{{z_r}}}}"


def transformed_row1_latex(names, classes: int) -> str:
    r"""``z = Θᵀu + θ0`` with ``u = φ(x)`` for a model linear in transformed features."""
    from ..feature_terms import feature_map_latex

    D = len(names)
    return (
        r"$$\mathbf{z}=\Theta^\top\mathbf{u}+\boldsymbol{\theta}_0,\quad "
        + feature_map_latex(names, maximum=3)
        + rf",\quad \Theta\in\mathbb{{R}}^{{{D}\times {classes}}},\quad K={classes}$$"
    )


def transformed_row3_latex(names, classes: int, definition: str) -> str:
    """Probability definition and the symbolic score in transformed features."""
    from ..feature_terms import symbolic_sum_latex

    terms = 3 if len(names) <= 3 else 2
    score = symbolic_sum_latex(names, lambda j: rf"\theta_{{{j},k}}", r"\theta_{0,k}", maximum=terms)
    return rf"$${definition},\quad z_k(\mathbf{{x}})={score},\quad k=0,\ldots,{classes - 1}$$"


def transformed_parameters_latex(weights, bias, max_cols: int, dec: int) -> str:
    r"""``Θ ∈ ℝ^{D×K}`` and ``θ0 ∈ ℝ^K`` with first/last rows and columns kept."""
    import numpy as np

    from ..feature_terms import visible_indices

    W = np.asarray(weights, dtype=float)
    b = np.asarray(bias, dtype=float).ravel()
    D, K = W.shape
    # Up to three rows fit the matrix slot; beyond that keep the first and the last.
    rows, rows_cut = visible_indices(D, 3 if D <= 3 else 2)
    cols, cols_cut = visible_indices(K, max(2, max_cols))

    def line(values):
        cells = [f"{values[c]:.{dec}f}" for c in cols]
        if cols_cut:
            cells = cells[:-1] + [r"\cdots"] + cells[-1:]
        return " & ".join(cells)

    lines = [line(W[r]) for r in rows]
    if rows_cut:
        width = len(cols) + (1 if cols_cut else 0)
        lines = lines[:-1] + [" & ".join([r"\vdots"] * width)] + lines[-1:]
    spec = "c" * (len(cols) + (1 if cols_cut else 0))
    matrix = r"\left[\begin{array}{" + spec + "}" + r"\\".join(lines) + r"\end{array}\right]"
    return (
        rf"$$\begin{{aligned}}\Theta&={matrix}\in\mathbb{{R}}^{{{D}\times {K}}}\\[5pt]"
        rf"\boldsymbol{{\theta}}_0&=\begin{{bmatrix}}{line(b)}\end{{bmatrix}}\in\mathbb{{R}}^{{{K}}}\end{{aligned}}$$"
    )


def transformed_scores_latex(weights, bias, names, dec: int) -> str:
    """First and last fitted class scores in transformed features."""
    import numpy as np

    from ..feature_terms import numeric_sum_latex

    W = np.asarray(weights, dtype=float)
    b = np.asarray(bias, dtype=float).ravel()
    K = W.shape[1]

    terms = 3 if len(names) <= 3 else 2  # two scores share one line

    def score(k):
        return numeric_sum_latex(W[:, k], b[k], names, dec, maximum=terms, signed=True)

    first = rf"z_0(\mathbf{{x}})={score(0)}"
    if K == 1:
        return rf"$${first}$$"
    separator = r"\;\cdots\; " if K > 2 else r"\qquad "
    return rf"$${first}{separator}z_{{{K - 1}}}(\mathbf{{x}})={score(K - 1)}$$"


def transformed_probability_latex(class_index: int, classes: int, probability_link: str) -> str:
    """Compact exact probability; the scores are written on the row above."""
    fraction = compact_probability_fraction_latex(class_index, classes, probability_link)
    return rf"$$\hat{{p}}(Y=c_{{{class_index}}}\mid\mathbf{{x}})={fraction}$$"


__all__ = [
    "MULTICLASS_ELLIPSIS_FONT_SIZE",
    "MULTICLASS_PROBABILITY_FONT_SIZE",
    "MULTICLASS_PROBABILITY_ROW_GAP",
    "compact_probability_fraction_latex",
]


def _with_fixed_height(fig):
    """Mark a figure whose equations sit at fixed paper heights.

    Size presets then change only its width (see ``design._apply_size``):
    scaling the height would pack the stacked equation rows into each other.
    """
    metadata = dict(fig.layout.meta or {}) if isinstance(fig.layout.meta, dict) else {}
    metadata["mlektic_layout"] = {**metadata.get("mlektic_layout", {}), "fixed_height": True}
    fig.update_layout(meta=metadata)
    return fig
