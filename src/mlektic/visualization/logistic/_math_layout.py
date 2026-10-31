"""Shared mathematical-layout tokens for logistic-regression figures."""

MULTICLASS_PROBABILITY_ROW_GAP = r"\\[10pt]"
MULTICLASS_PROBABILITY_FONT_SIZE = 13
MULTICLASS_ELLIPSIS_FONT_SIZE = 22


def compact_probability_fraction_latex(class_index: int, classes: int, probability_link: str) -> str:
    """Return an exact compact multiclass probability fraction.

    The expanded score remains visible separately.  This form is used only in
    constrained layouts where repeating every numeric score in the denominator
    would cross into an adjacent plot.
    """
    if probability_link == "ovr":
        return rf"\frac{{q_{{{class_index}}}}}{{\sum_{{\ell=0}}^{{K-1}}q_\ell}},\quad q_\ell=\sigma(z_\ell)"
    return rf"\frac{{e^{{z_{{{class_index}}}}}}}{{\sum_{{\ell=0}}^{{K-1}}e^{{z_\ell}}}}"


__all__ = [
    "MULTICLASS_ELLIPSIS_FONT_SIZE",
    "MULTICLASS_PROBABILITY_FONT_SIZE",
    "MULTICLASS_PROBABILITY_ROW_GAP",
    "compact_probability_fraction_latex",
]
