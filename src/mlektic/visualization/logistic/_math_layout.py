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
