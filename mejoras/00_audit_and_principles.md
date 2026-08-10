# Audit and design principles

## Overall assessment

Mlektic already has a strong differentiator: it connects fitted estimators to animated mathematical explanations instead of offering generic diagnostic charts. The neural family is currently the clearest expression of that idea. Its architecture, training, and mathematical report views feel more deliberate, modular, and academically framed than the tabular family.

Linear and logistic regression are functional and cover unusually broad configurations, but their pedagogical contract is not yet as explicit. Several visual states can be mistaken for genuine optimizer history, temporal sampling hides original indices, smoothed loss can disagree with a metric card, and some invalid configuration values are silently accepted. The highest-priority work is therefore semantic rather than cosmetic.

## Verified technical evidence

### History provenance

- Incremental Scikit-learn estimators are cloned and refitted to construct an animation. The animation is a replay, not a recording of the original `fit` call.
- Replay configuration changes effective estimator parameters such as `max_iter`, `tol`, `warm_start`, and `shuffle` when supported.
- A replay can finish at parameters that differ materially from the already fitted estimator.
- Non-incremental estimators use a baseline-to-final interpolation. Its final state matches the fitted estimator, but intermediate states are synthetic and are not optimizer updates.
- Earlier figures used the generic word “Step” for both mechanisms, which made this distinction invisible.

### Temporal sampling

- `steps` constructs a semantic history of length K.
- `max_frames` or `frame_step` then reduces it to N visible checkpoints.
- Before phase 0, the retained frames were relabeled `0..N-1`, losing their coordinates in the K-state source history.
- Hybrid linear animation additionally creates perceptual subframes between semantic checkpoints. These subframes improve motion but must not be presented as training updates.

### Smoothing

- EMA smoothing was applied after metric histories were created.
- The loss trace could therefore be smoothed while the metric card displayed the unsmoothed value.
- The raw series was overwritten through the backward-compatible `loss_hist` key, preventing a direct audit.

### Layout and styling

- Tabular figures use the classic dark theme, a fixed 1100 × 600 base size, `autosize=False`, large titles, and relatively generous top mathematical annotations.
- Neural figures use more compact typography, spacing, and panel composition. They generally create a clearer academic hierarchy.
- A smaller type scale and tighter spacing would benefit tabular figures, especially in notebooks, but changing defaults now would create unnecessary visual regressions.
- The correct direction is a tokenized optional design system: preserve `classic`, then add `academic`, `compact`, `classroom`, and `accessible` presets.

### Prediction explanations

- A supplied `yhat`, `p_hat`, or `y_hat` could disagree with the estimator without warning.
- Linear prediction accepted multiple query rows but visualized only the first one.
- Some 1D and 2D plot ranges clipped a query outside the training range.
- Logistic binary formulas assumed integer labels, even when Scikit-learn supports string labels.
- Extrapolation was not identified explicitly.

### Export

- Plain `fig.write_html(...)` can leave mathematical annotations as raw LaTeX when MathJax is not loaded.
- The existing optimized notebook renderer included MathJax, but the public export path was not centralized or documented.
- Plotly can inline its own JavaScript runtime, but its public HTML API does not provide a supported self-contained MathJax bundle. A MathJax CDN export must therefore be described as network-dependent.

### Notebooks

- The four notebooks exercise many valuable variants and have served as exploratory QA.
- They mix experimentation, visual snapshots, debugging, and pedagogy in large files with heavy stored output.
- Coverage is broad but implicit: there is no generated case matrix, assertion layer, expected-result index, or lightweight smoke notebook.
- The notebooks should be preserved as evidence while new focused QA and student notebooks are introduced.

## Governing principles

### 1. Truth before beauty

Every visual state must identify what produced it. The terms `recorded`, `replayed`, and `interpolated` are not interchangeable. A synthetic path may be pedagogically valuable, but only when it is labeled as such.

### 2. Motion is part of the explanation

Animation supports visual learners by showing continuity, geometry, and parameter effects. Performance work should reduce redundant semantic checkpoints or generate trace-only visual subframes; it must not remove movement without an explicit user choice.

### 3. One mathematical grammar across model families

Every family should progressively expose:

1. data and variables;
2. model equation;
3. parameter values and feature-space convention;
4. substitution for a concrete observation;
5. link or activation function;
6. objective, metric, and regularization where applicable;
7. prediction or class decision;
8. provenance and temporal semantics.

### 4. Separate objective, displayed loss, and metrics

An optimization objective, an empirical loss trace, a regularization term, and an evaluation metric are different quantities. They require distinct names and, when applicable, distinct raw and display values.

### 5. Progressive information density

The default should remain approachable. Formal depth should be available through structured density levels rather than a single overloaded figure. Proposed levels are `essential`, `academic`, and `complete`.

### 6. Explicit compatibility

New themes, formats, sizes, and density controls are additive. `classic` and current dimensions remain the baseline until a versioned migration is deliberately approved.

### 7. Accessibility is a design constraint

Color cannot be the only carrier of meaning. Text must remain readable at notebook widths, controls must be identifiable, and motion should eventually support reduced-motion behavior without eliminating the default animated experience.

## Official temporal vocabulary

- **T — training updates:** updates reported or recorded by the training process, when known.
- **K — semantic checkpoints:** recorded, reconstructed, or constructed model states available to the visualization.
- **N — displayed checkpoints:** states retained after temporal sampling, where `N ≤ K`.
- **q — perceptual intervals:** visual intervals inserted between adjacent displayed checkpoints.
- **F — visual frames:** rendered frames. For a simple hybrid path, `F = (N - 1)q + 1`.

The figure should show N/K when reduction occurs. If T is known and semantically meaningful, it should be shown separately. Interpolation should use progress or α, never “epoch” or an unqualified “training step.”

## Visual reference

The neural family is the internal aesthetic benchmark: compact hierarchy, restrained typography, intentional panels, and mathematical content organized as a report. Tabular figures should converge toward that discipline without copying neural layouts where geometry requires a different form.

## Decisions frozen for early phases

- Do not change classic colors, default dimensions, default animation mode, or public function names in phase 0.
- Do not append a fitted estimator state to a replay and call it another optimizer step.
- Do not smooth stored empirical data in place.
- Do not renumber retained checkpoints as though they were consecutive source states.
- Do not silently coerce unknown modes, themes, baselines, display spaces, links, or metric names.
- Do not describe a CDN-dependent artifact as fully offline.
