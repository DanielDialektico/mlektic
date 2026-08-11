# Phase 3 implementation record — additive visual design system

## Status

Phase 3 is implemented on `feature/phase-3-visual-design`. This record explains
what changed, what deliberately did not change, how every public option is
resolved, and which constraints remain. It is the durable implementation note
for future maintenance and review.

## Compatibility decision

The classic dashboard is still the default. A call that omits every Phase 3
argument keeps the existing fixed dimensions, dark palette, subplot geometry,
trace widths, equations, controls, frame count, frame duration, and slider
semantics. The only addition is a machine-readable `mlektic_visual` entry in
`layout.meta` and a stable `uirevision` key.

No theme, format, or size option changes history construction. In particular:

- `steps`, `max_frames`, and `frame_step` still determine semantic retention;
- `frame_duration`, `transition_duration`, `fps`, and
  `interpolation_frames` still determine playback cadence;
- smoothing still affects only its documented display series;
- `dashboard`, `compact`, and `lesson` preserve all retained frames;
- only `format="report"` and `reduced_motion=True` intentionally freeze the
  exact last displayed state and remove playback controls.

## Public visual contract

Training visualizers expose the following independent arguments:

```python
visualize_lr(
    ...,
    theme=None,
    format="dashboard",
    density=None,
    size="default",
    width=None,
    height=None,
    responsive=False,
    reduced_motion=False,
)
```

`visualize_logistic` has the same contract. Prediction explainers and neural
views expose the applicable visual axes as well.

### Theme

Theme controls typography, color, line weight, panel color, controls, and
accessibility redundancy. Registered themes are:

- `classic`: the compatibility baseline;
- `academic`: restrained typography and report-like contrast;
- `classroom`: larger type, controls, and geometry for projection;
- `compact`: reduced type and spacing for comparisons;
- `accessible`: a color-vision-safe palette plus marker and dash redundancy.

The token source is `visualization/design.py`. `VisualTokens` is immutable and
public through `get_theme_tokens()`, so a notebook can inspect exactly which
values were resolved. No web font is downloaded by Mlektic; font-family values
contain local fallbacks.

### Format

Format controls composition, not color or mathematical truth:

- `dashboard`: the existing combined animated view;
- `lesson`: adds Data, Model, Objective, and Complete stages while retaining
  the original Play/Pause controls and every frame;
- `compact`: reduces canvas height and margins without decimating motion;
- `report`: applies the last frame exactly, removes temporal controls, and
  reclaims the slider area for a static academic artifact.

The lesson stages change Plotly trace visibility. They do not recompute or
discard any trace. A learner can reveal the full composition at any time.

### Density

For linear and logistic training views, `density` is a compatible alias for
the Phase 1 `detail` contract:

- `essential`: main animated mathematics and machine-readable contract;
- `academic`: fitted-model derivation and empirical evaluation;
- `complete`: preprocessing, objective, regularization, and optimizer caveats.

When `density` is omitted, `detail` remains authoritative. Supplying
inconsistent non-default values raises an explicit error rather than silently
choosing one. Prediction and neural figures record the selected density for
cross-family composition, but they retain their established mathematical
panel semantics.

### Size and responsiveness

Named sizes are `default`, `compact`, `notebook`, `wide`, and `classroom`.
Explicit `width` and `height` values override the preset. Dimensions must be
integer pixel values of at least 320.

`responsive=True` sets Plotly `autosize`, removes a preset width when no
explicit width was requested, and records the responsive export configuration.
It scales a resolved composition; it does not pretend Plotly subplots support
CSS reflow. Structural rearrangement remains the responsibility of `format`.
This distinction is stated in figure metadata.

`export_figure(..., responsive=None)` now inherits the figure contract. A
classic figure without responsive metadata still exports fixed-size, preserving
the previous behavior. An explicit export value always wins.

## Processing order

The public orchestration is intentionally deterministic:

1. validate and resolve the visual specification;
2. build history and the estimator-backed mathematical contract;
3. construct the existing Plotly figure and frames;
4. configure animation and provenance labels;
5. attach the mathematical panel;
6. apply theme, accessibility, format, size, and responsive post-processing;
7. attach stable notebook control styling.

This order prevents design choices from contaminating model state, evaluation,
or temporal provenance.

## Metadata contract

Every public Phase 3 figure records `layout.meta["mlektic_visual"]` with:

- schema version, family, theme, format, density, and size;
- requested and resolved dimensions;
- responsive and reduced-motion choices;
- every resolved token;
- whether motion is preserved;
- export configuration;
- the scaling-versus-reflow statement;
- accessibility and static-alternative declarations.

Existing `mlektic_history`, `mlektic_math`, and `mlektic_prediction` entries are
preserved. The visual post-processor merges metadata rather than replacing it.

## Accessibility decisions

The accessible theme uses both color and non-color encodings:

- data markers use open symbols;
- multiclass traces receive distinct marker symbols;
- objective paths use a dotted line;
- decision boundaries use a dashed line;
- line widths and text contrast increase;
- `report` and `reduced_motion` provide exact static alternatives.

Prediction explainers also place plotted `y_hat` and `p_hat` values inside a
theme-aware opaque panel with a contrasting border. This preserves the arrow's
geometric link to the predicted point while preventing the number from
disappearing into observations, fitted lines, surfaces, or grid lines.

Plotly's built-in controls remain keyboard-focusable where supported by its
renderer. Mlektic does not claim full WCAG conformance because MathJax, Plotly,
notebook hosts, and exported HTML each contribute their own accessibility
behavior.

## Visual QA and automated evidence

Automated Phase 3 tests cover:

- the public immutable token registry and invalid-option errors;
- classic dimensions, colors, geometry, and frame retention;
- density/detail compatibility;
- all four formats;
- exact final-frame freezing;
- accessible marker and line redundancy;
- named sizes and explicit 700, 1000, and 1400 pixel widths;
- responsive metadata and HTML export inheritance;
- linear and logistic prediction explainers;
- a public neural architecture view;
- JSON serialization of the visual contract.

Manual rendered-image inspection covered classic, academic, compact,
classroom lesson, academic report, and accessible logistic figures. It found
and corrected a report-only title/equation collision by preserving the original
top mathematical band while removing the unused slider region. Notebook review
also exposed a compact-plus-academic collision: shrinking the canvas and the
notebook size preset had reduced the plot domain enough to pull negative-paper
panel coordinates into the slider. Compact layouts now preserve the lower-panel
reserve and shift animated upper equations consistently when the header is
reduced.

## Known boundaries

- Responsive mode scales the chosen composition; narrow structural reflow
  requires selecting `compact`, `lesson`, or `report`.
- A 700-pixel two-panel dashboard is usable but intentionally dense. The
  compact format is recommended, and report/lesson are preferable when long
  equations or many classes are involved.
- Accessibility redundancy is strongest for 2D scatter/line views. Plotly 3D
  surfaces still depend partly on hue and spatial geometry, accompanied by
  textual legends and mathematical labels.
- Themes post-process neural figures so the established neural layout remains
  the classic baseline; Phase 3 does not force tabular subplot geometry onto
  neural architecture views.

## End-user verification artifact

`phase_3_visual_design.ipynb` is the executable user-facing showcase. It uses
real fitted estimators and renders real figures without unit-test assertions.
It demonstrates compatibility, independent visual axes, motion preservation,
lesson staging, accessible encodings, report output, responsive sizing,
prediction explainers, and metadata inspection.
