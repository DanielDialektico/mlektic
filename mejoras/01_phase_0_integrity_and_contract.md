# Phase 0 — Mathematical integrity and history contract

## Objective

Make every linear and logistic figure truthful, auditable, and deterministic enough to support later mathematical and visual improvements. Phase 0 does not redesign figures. It establishes the contracts that later phases can trust.

## Scope

### A. Provenance contract

Every tabular history payload must contain a `metadata` mapping with this stable conceptual schema:

```python
history["metadata"] = {
    "schema_version": 1,
    "source": "recorded" | "replayed" | "interpolated",
    "source_detail": {...},
    "requested_mode": str,
    "resolved_mode": "iterative" | "final_interp",
    "requested_steps": int,
    "training_total_steps": int | None,
    "captured_steps": int,
    "displayed_steps": int,
    "step_indices": np.ndarray,
    "displayed_step_indices": np.ndarray,
    "final_state_matches_estimator": bool | None,
    "display_space": "original" | "scaled",
    "smoothing": {"method": None | "ema", "beta": float},
    "decimation": {"max_frames": int | None, "frame_step": int | None},
    "warnings": list[dict],
}
```

Existing payload keys remain available. `history_kind`, `loss_hist`, coefficient histories, grids, and probability histories are not removed in this phase.

### B. Strategy semantics

#### `recorded`

Reserved for states captured during the actual training process. The tabular Scikit-learn implementation does not claim this source in phase 0 because the library currently receives an already fitted estimator.

#### `replayed`

Used when Mlektic clones an incremental estimator and reconstructs a path. Required disclosures:

- it is not the original `fit` call;
- the estimator class;
- effective replay overrides;
- relevant original parameters when available;
- whether the final replay parameters match the supplied fitted estimator;
- a warning when they do not match.

The original fitted state must not be silently appended as if it were the next replay update. A later phase may display it as a distinct reference state.

#### `interpolated`

Used for baseline-to-fitted-model paths. Required disclosures:

- states are synthetic;
- the path is not optimizer history;
- the interpolation coordinate α;
- the baseline convention;
- whether the final state matches the fitted estimator.

### C. Temporal coordinates

Strategies create source coordinates before decimation:

- replay checkpoints use `1..K`, reflecting constructed fit/update states;
- interpolation keeps source indices and `alpha_values` from 0 to 1;
- decimation samples every time-aligned array using the same retained positions;
- full source indices remain in metadata;
- retained indices remain at the top level and in `displayed_step_indices`.

Figure frame names remain internal Plotly positions for compatibility. User-facing slider labels and any loss-axis tick labels show the retained source coordinates. Interpolation uses percentages or α and must not use the generic prefix “Step.”

### D. Raw and display loss

The payload contains:

```python
history["loss_raw"]      # empirical values before display smoothing
history["loss_display"]  # values used by the figure
history["loss_hist"]     # backward-compatible alias of loss_display
```

EMA reads `loss_raw` and writes a separate array. If a visible metric card is labeled `Loss` or `Log-loss`, it uses the same display series as the curve. Other metrics continue to use values computed from the model checkpoints. Metadata always states the smoothing method and β.

### E. Central validation

Configuration construction fails early with an English error message for:

- unsupported `mode`, `baseline`, `display_space`, `smooth`, or `multiclass_link`;
- non-positive `steps`, grid sizes, frame budgets, or sampling strides;
- `smooth_beta` outside `[0, 1)`;
- unknown built-in metric names;
- non-callable custom metrics;
- unknown themes;
- `mode="iterative"` on an estimator without `partial_fit`;
- empty, non-finite, mismatched, or unsupported multi-output training data.

The legacy spelling `smooth="none"` receives a deprecation warning and becomes `None`. No other unknown value silently falls back.

### F. Prediction integrity

Prediction explainers must:

- require exactly one query sample;
- validate feature count and finite values;
- compute the estimator prediction and probability even when a display value is supplied;
- verify supplied values by default with documented tolerances;
- require `prediction_source="provided"` for an intentional counterfactual;
- validate probability bounds, vector length, normalization, and fitted class identity;
- support non-integer class labels in formulas;
- mark features outside their observed training range;
- expand 1D and 2D ranges so the query remains visible;
- include provenance and range information in figure metadata.

This policy prevents a visually authoritative explanation from silently presenting a value the model did not produce.

### G. Mathematical HTML export

Provide one public helper that controls:

- full HTML vs notebook fragment behavior;
- inline vs CDN Plotly runtime;
- explicit MathJax inclusion;
- autoplay;
- responsive configuration;
- UTF-8 output and destination extension.

The current reliable default is inline Plotly plus MathJax CDN. This is not fully offline because equation rendering requires a network request. If MathJax is disabled, the artifact must be understood as preserving LaTeX source rather than guaranteeing rendered mathematics.

## Compatibility rules

- Classic theme remains the default.
- Default width, height, spacing, trace style, and motion remain unchanged.
- `loss_hist` remains available, now with clearly documented display semantics.
- Existing animation frames and internal names remain compatible.
- New title subtitles and slider labels provide transparency without changing geometry.
- New validation can reject configurations that were previously ignored; this is an intentional correctness improvement and must be listed in release notes.

## Required tests

### Invariants

- Interpolation ends at α = 1 and matches the fitted estimator when coefficients are extractable.
- Replay never claims to be recorded training.
- A mismatched replay final state is represented as `False`, not hidden.
- Source and retained indices start and end at the correct coordinates.
- All time-aligned arrays have the displayed length after decimation.
- `loss_raw` remains unchanged after smoothing.
- the loss curve and visible loss metric use `loss_display`.

### Validation

- Every supported enum rejects an unknown value.
- Numeric controls reject booleans, non-numeric values, invalid ranges, and zero where positive values are required.
- Unknown metric names do not disappear silently.
- A non-incremental estimator rejects explicit replay mode.

### Prediction

- Supplied correct values pass model verification.
- Supplied incorrect values fail by default.
- Explicit counterfactual mode succeeds and is labeled.
- Multiple linear queries fail instead of truncating.
- String class labels render without integer conversion.
- Out-of-range queries are visible and marked as extrapolation.

### Export

- The helper writes UTF-8 full HTML.
- The chosen Plotly dependency strategy is reflected in the output.
- MathJax CDN output contains the MathJax loader.
- invalid dependency modes and extensions fail clearly.

## Acceptance criteria

Phase 0 is complete when:

1. no tabular history is described ambiguously as training history;
2. a student can distinguish replay from interpolation in the figure itself;
3. N/K and retained source coordinates survive decimation;
4. smoothing is auditable and internally consistent;
5. prediction explainers cannot silently contradict the estimator;
6. HTML mathematical dependency behavior is explicit;
7. classic visual and motion defaults remain intact;
8. all new contracts are documented in English;
9. unit tests, Ruff, and Sphinx complete successfully.

## Risks and mitigations

- **Longer titles:** use a visually secondary subtitle and optimize typography in phase 3.
- **More exceptions:** document accepted values and use precise remediation messages.
- **Replay mismatch surprises users:** treat the surprise as important model-literacy information.
- **Metadata serialization:** keep values JSON-oriented where practical; NumPy timeline arrays remain consistent with existing numeric payloads.
- **CDN misunderstanding:** describe MathJax network dependence in the docstring, guide, and implementation record.

## Deliverable

The deliverable is not a redesigned figure. It is a trustworthy foundation: validated inputs, explicit provenance, preserved time, auditable smoothing, verified prediction explanations, deterministic export semantics, tests, and English documentation.
