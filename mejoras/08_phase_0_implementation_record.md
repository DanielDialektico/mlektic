# Phase 0 implementation record

## Status

Implemented on the working branch. This record describes the behavior delivered in phase 0, the design decisions behind it, compatibility consequences, verification evidence, and known limitations that remain for later phases.

## Implemented behavior

### Validated configuration

`LinearHistoryConfig` and `LogisticHistoryConfig` now validate all public history fields at construction time:

- mode, smoothing, baseline, display space, and multiclass link enums;
- positive steps, grid sizes, frame budgets, and frame strides;
- EMA β in `[0, 1)`;
- built-in metric names and custom callable mappings.

Unknown values no longer fall back silently. The legacy `smooth="none"` spelling emits a deprecation warning and resolves to `None`.

The history engine also validates numeric finite feature matrices, target length, supported one-dimensional targets, and explicit iterative mode capability. This makes failure occur before a partially constructed figure exists.

### Provenance

Tabular strategies now emit:

- `history_source="replayed"` for a cloned incremental reconstruction;
- `history_source="interpolated"` for a baseline-to-final path;
- source coordinates for every constructed state;
- `alpha_values` for interpolation.

The engine attaches schema-versioned metadata with estimator/source detail, requested/captured/displayed counts, full and retained coordinates, estimator-reported iterations, display space, smoothing, decimation, warnings, and a final-parameter comparison where possible.

The `recorded` value is documented but intentionally unused for tabular Scikit-learn estimators because Mlektic receives an already fitted model and did not observe its original `fit` call.

### Final-state comparison

For extractable linear and logistic coefficients, the last constructed learned-space parameters are compared with the supplied fitted estimator using `rtol=1e-7` and `atol=1e-9`.

- interpolation should normally report `True`;
- replay can report `True` or `False` depending on estimator and fit configuration;
- unavailable parameters report `None` rather than an invented result.

Phase 0 did not append a fitted state to a replay path; a mismatch remained
visible as evidence. Phase 1 later superseded the presentation policy by
reserving a distinctly labeled `fitted` endpoint while preserving every
retained state's origin. It is not described as another replay update.

### Temporal preservation

Replay source checkpoints use `1..K`. Interpolation stores a source index plus α from 0 to 1. Decimation applies one retained-index vector to every aligned NumPy history and metric array. Metadata keeps the original K coordinates; the top-level arrays and `displayed_step_indices` keep the retained N coordinates.

The public linear and logistic visualization functions add a secondary title line and relabel sliders:

- replay: `Replay checkpoint` plus retained source indices;
- interpolation: `Interpolation progress` plus percentage labels.

Loss axes titled `Step` are relabeled consistently. Internal Plotly frame names are unchanged, preserving current animation construction.

### Raw vs display loss

History payloads now expose:

- `loss_raw` — empirical captured/constructed values;
- `loss_display` — raw or EMA values used visually;
- `loss_hist` — a backward-compatible alias of `loss_display`.

Smoothing creates a new array. The visible `Loss`/`Log-loss` metric series is updated from the same display array, removing the prior line/card disagreement. Metadata records smoothing method and β even when no smoothing is used.

### Prediction integrity

Linear and logistic prediction explainers now:

- accept exactly one finite query;
- always compute estimator output;
- verify supplied values by default;
- require `prediction_source="provided"` for intentional counterfactuals;
- validate probability bounds, normalization, and class identity;
- support string class labels in binary formulas;
- expose both class probabilities and the winning class index in prediction
  explanations, while fitted semantic labels remain optional through
  ``show_class_labels=False`` and are always retained in figure metadata;
- map binary string targets to numeric 0/1 geometry in prediction plots so
  Plotly cannot convert the probability axis to a categorical axis and hide
  the model curve or surface;
- render the two-feature decision boundary as the line where the probability
  surface crosses 0.5 rather than as a full horizontal threshold plane;
- identify out-of-range features;
- expand relevant 1D/2D plot ranges to include the query;
- write source, model/display values, feature-range state, and link semantics into figure metadata.

The figure subtitle says whether the displayed value is model-verified or user-provided and whether the query is an extrapolation.

### Export

The new public `export_figure` helper writes UTF-8 full HTML and validates:

- `include_plotly="inline" | "cdn"`;
- `include_mathjax="cdn" | False`;
- responsive/autoplay booleans;
- `.html`/`.htm` destination semantics.

Default behavior inlines Plotly and loads MathJax from its CDN. This creates a self-contained Plotly runtime but not a fully offline mathematical renderer. The limitation is explicit in the public docstring and this record. `show_optimized` was translated to English and imports IPython lazily so importing Mlektic does not require notebook dependencies.

## Compatibility assessment

Preserved:

- classic theme remains the default;
- base width, height, autosize behavior, colors, line widths, and spacing remain unchanged;
- current native/hybrid motion and interpolation remain unchanged;
- public history keys remain available;
- internal frame names and trace geometry remain unchanged.

Intentional observable changes:

- history figures have a provenance/timeline subtitle;
- slider and loss-axis labels use honest source coordinates;
- invalid values that were previously ignored now raise;
- inconsistent supplied prediction values now raise unless explicitly counterfactual;
- raw loss is available separately and the visible loss metric agrees with the smoothed curve.

## Verification evidence

The phase adds contract tests covering:

- interpolation provenance, α endpoints, final match, and decimation;
- replay provenance, retained indices, warnings, and raw/display loss;
- invalid configuration values and non-incremental replay rejection;
- visible source subtitle and slider labels;
- linear mismatch, counterfactual, extrapolation, and multi-query rejection;
- logistic mismatch and string labels;
- logistic smoothing consistency;
- Plotly/MathJax export dependency declarations.

At implementation time:

- 68 tests pass, including binary string-label visualization and F1 semantics;
- Ruff passes after import normalization;
- existing visual API tests remain green;
- Sphinx builds successfully in English with warnings treated as errors.

## Known limitations

- replay remains batch replay over a clone; it is not a recorder for the original fit call;
- the first replay state follows a one-iteration clone fit, not an untrained zero state;
- estimator `n_iter_` is informative but not universally comparable with replay checkpoints;
- metadata contains NumPy arrays because numeric history payloads already use NumPy; direct JSON consumers must convert them;
- only the classic theme exists in phase 0;
- fixed sizing remains the default and responsive layout/reflow belongs to phase 3;
- dense multiclass 2D headers leave little room for a provenance subtitle; the slider therefore repeats source and N/K so the contract stays visible until phase-3 spacing work;
- MathJax CDN export requires network access;
- exact optimizer-objective introspection and full regularization pedagogy belong to phase 1;
- non-affine preprocessing cannot yet be expressed as raw-space coefficients;
- neural recorder metadata has not yet been unified with the tabular schema.

## Follow-up gates

Before phase 1 begins:

1. inspect representative replay/interpolation HTML figures manually;
2. confirm the title subtitle does not collide in 1D, 2D, nD, and multiclass routes;
3. decide whether the metadata schema should use NumPy arrays or JSON-native lists before a public 1.0 contract;
4. decide the release/version boundary for the accumulated changelog.

## Post-review mathematical layout safeguards

The multiclass probability stack now uses shared mathematical-layout tokens in
the 1D, 2D, and nD builders. The representative-class substitution keeps its
existing annotation anchor, while a 10-point LaTeX row gap lowers only the
expanded probability fraction relative to the preceding score equation. This
balances the whitespace above and below the vertical ellipsis without moving
the complete mathematical block or changing animation geometry.

The first and final probability fractions use the same 13-point typography,
and the ellipsis uses a shared 22-point size in every applicable builder.
Contract tests compare the complete stack-position and typography signature
against every animation frame so future formula or history changes cannot
silently introduce vertical drift.

The current full validation contains 78 passing tests, and Ruff reports no
issues in the maintained logistic visualization modules or the phase-0
contract tests.

## Cross-configuration layout stabilization

A rendered audit covered interpolation and replay figures for linear, binary
logistic, and multiclass logistic models in 1D, 2D, and nD, including
``K=8, d=20`` stress cases. The audit found no frame-to-frame movement of axes,
controls, domains, or margins, but it exposed configuration-specific static
collisions that coordinate-only tests could not detect.

The stabilization preserves the classic default dimensions and animation
behavior while applying the following internal layout rules:

- two-feature linear prediction outputs use 15-point inline coordinates for
  ordinary values and fall back to a 13-point wrapped form only for genuinely
  long formatted coordinates;
- the 2D multiclass coefficient matrix and bias vector are independent
  annotations; the bias is vertically centered between the matrix and the
  class-score equation;
- expanded numeric probability fractions remain the default, while replay
  layouts with a visible loss panel use the exact compact forms
  ``q_k / sum_j q_j`` or ``exp(z_k) / sum_j exp(z_j)`` after displaying the
  expanded numeric score separately;
- 1D and nD ellipses are centered between their first and final probability
  examples, with frame-stable anchors and shared typography;
- compact nD input matrices allocate only the rows required by the visible
  features instead of padding to the maximum capacity;
- dense nD multiclass layouts use a taller canvas and honor
  ``max_theta_cols`` so capped parameter and bias displays do not enter the
  loss panel;
- linear and binary-logistic nD routes construct a single mathematical panel
  when ``show_loss=False`` instead of retaining an empty loss subplot;
- provenance subtitles retain source, N/K, estimator ``n_iter_``, and endpoint
  information in a shorter form suitable for fixed-width notebooks. Phase 1
  subsequently replaced visible mismatch endpoints with a distinct fitted
  reference state.

Regression tests cover the independent 2D parameter blocks, compact replay
fractions, ellipsis geometry, dynamic input-vector height, dense matrix caps,
single-panel no-loss layouts, and frame-level annotation stability.
