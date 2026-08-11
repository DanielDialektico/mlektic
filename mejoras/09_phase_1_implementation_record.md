# Phase 1 implementation record

## Status

Implemented on `feature/phase-1-mathematical-parity`. This record describes the
delivered mathematical contract, compatibility decisions, exactness boundaries,
test evidence, notebook coverage, and work intentionally deferred to later
phases.

## Compatibility decision

The existing visualization is the public baseline.

- `detail="essential"` is the default.
- Essential detail retains the compact canvas, classic theme, controls,
  semantic frames, hybrid subframes, frame duration, transition duration, and
  slider behavior. The one deliberate shared-layout correction is documented
  below: the evolving one-dimensional equation moved out of the data axes.
- Mathematical metadata is added under `layout.meta["mlektic_math"]`; the
  metadata itself does not alter rendering.
- `detail="academic"` and `detail="complete"` are explicit opt-in extensions.
- Academic panels increase height and bottom margin only in those opt-in modes.
- The academic panel is a stable fitted-model reference. It does not update on
  every visual subframe and therefore does not force hybrid animations to
  redraw MathJax layouts.

This distinction is visible in the phrase **Fitted-model derivation**. The
existing title/slider continue to identify whether the animated states are a
replay or a synthetic interpolation.

### Shared evolving-equation standard

The one-dimensional hybrid view previously represented its evolving numeric
equation as an HTML text trace positioned above the observations in the data
axis. This was fluid, but it mixed mathematical exposition with model geometry
and made the academic view appear less rigorous than the native view.

The public `visualize_lr()` path now uses the same composition at every detail
level:

- the symbolic model definition remains in the upper mathematical area;
- the fitted numeric equation evolves in LaTeX in a reserved, axis-independent
  math band;
- the data and model line use only the data region;
- loss retains its own axis and metrics retain their ordered card column;
- `academic` and `complete` add only the separate panel below the slider.

The LaTeX equation is still a trace, not a per-frame layout annotation. Its
coefficients use the same continuous interpolation positions as the model line,
loss, and metrics. Frames remain layout-empty and playback retains
`redraw=False`. The low-level builder keeps its classic placement default for
direct compatibility; the public API selects the standardized math band.

## Public API additions

Both tabular visualization APIs now accept:

```python
detail="essential"               # essential | academic | complete
show_objective="auto"            # auto | bool
show_regularization="auto"       # auto | bool
feature_names=None                # original feature names
sample_index=None                 # deterministic default: first row
```

`visualize_logistic()` additionally accepts:

```python
threshold=0.5                     # binary decision threshold, 0 < tau < 1
class_focus=None                  # fitted label or zero-based index
```

Validation occurs before the final figure is returned. Invalid detail values,
visibility controls, thresholds, feature-name counts, sample indexes, and class
focus values raise actionable English errors.

## Shared mathematical contract

Every tabular figure exposes a versioned contract at
`figure.layout.meta["mlektic_math"]`. The contract contains:

- family and final estimator type;
- requested and actual equation space;
- original and transformed feature dimensions;
- preprocessing step names and types;
- verified affine map, when available;
- original or transformed feature names;
- coefficients/weight matrix and separate intercept/bias;
- selected observation and observation-specific contributions;
- reconstructed and estimator predictions/probabilities;
- a strict numerical match flag;
- exact empirical objective name, normalization, formula, value, and role;
- evaluation metrics where applicable;
- fitted class order, positive class, threshold, probability link, focus, and
  winning class;
- public regularization family and strength settings;
- conservative intercept-penalty and normalization statements;
- canonical update equation and a flag that prevents it from being described
  as the estimator's exact private rule;
- replay/interpolation source.

The visible panel is generated from this contract. It is not a second,
independent calculation.

## Linear regression

The selected observation is reconstructed as

\[
\hat y_i=\theta_0+\sum_j\theta_jx_{ij}.
\]

Metadata stores every `theta_j * x_ij` contribution separately from the
coefficient. Their sum plus the intercept must match `estimator.predict()`.
The visible panel wraps named coefficient-value products into LaTeX rows of at
most three terms. It shows every contribution through nine features. Above
nine, it selects the nine largest absolute observation-specific contributions,
adds a separate disclosure of the visible and total counts, and keeps the
complete ordered vector in metadata. The panel height and annotation spacing
grow from the actual mathematical row count, preventing both horizontal
clipping and vertical collisions.

MSE uses the explicit `1/n` convention. MAE and R-squared are labeled as
evaluation metrics. `LinearRegression` is described as minimizing residual sum
of squares, with the precise note that MSE differs by a positive constant and
has the same minimizer. Other estimators do not receive an inferred private
normalization.

## Binary logistic regression

The contract reconstructs

\[
z_i=\theta_0+\mathbf{x}_i^\top\boldsymbol{\theta},\qquad
p_i=\sigma(z_i),\qquad
\hat y_i=\mathbb{1}[p_i\ge\tau]
\]

in fitted class-index notation. `classes_[1]` is recorded as the positive
class. Semantic labels remain hidden by default, matching the Phase-0 decision
to avoid unexplained domain words such as "accepted" and "rejected". Setting
`show_class_labels=True` reveals them without changing fitted ordering.

A non-default `threshold` updates the indexed probability axis and academic
threshold geometry. The contract explicitly calls this a user threshold
applied to model probability; it does not claim that `estimator.predict()` used
that non-default threshold.

Binary log-loss maps arbitrary fitted labels through
`y_i' = I[y_i == classes_[1]]`.

## Multiclass logistic regression

Scores use one fitted class column per `classes_` entry:

\[
z_k(\mathbf{x})=b_k+\mathbf{x}^\top\mathbf{w}_k.
\]

The resolved Softmax or normalized-OvR link is stored and used to reconstruct
the complete probability vector. The reconstructed vector must match
`predict_proba()`. The decision is the argmax index in fitted class order.

`class_focus` reduces visual competition in 1D/2D by keeping one curve or
surface visible. The title says `class focus c_k (1/K)`. All class parameters,
probabilities, and labels remain in metadata. No traces or frame states are
deleted; non-focused class traces are only marked invisible and can still be
inspected programmatically.

## Pipelines and feature spaces

The fitted preprocessing map is evaluated on zero, basis vectors, and the
supplied data. If the transformation is numerically affine,

\[
\mathbf{u}=A\mathbf{x}+\mathbf{c},
\]

Mlektic verifies the map and converts model-space parameters by

\[
\mathbf{w}_o=A^\top\mathbf{w}_m,
\qquad
b_o=b_m+\mathbf{c}^\top\mathbf{w}_m.
\]

The converted calculation must reproduce the pipeline prediction. This is more
general than relying only on a hard-coded standard-scaler name while retaining
the existing standard-scaler history behavior.

When preprocessing is non-affine, raw-space coefficients are not claimed. The
contract switches to transformed-feature mathematics and uses
`get_feature_names_out()` where available. Polynomial pipelines therefore show
features such as `x_1` and `x_1^2`, not a fictitious raw linear coefficient.
Such a model remains linear in its fitted coefficients and transformed
features while being nonlinear in the original input. For
`phi(x) = (x, x^2)`, the displayed original-space geometry is the parabola
`theta_0 + theta_1*x + theta_2*x^2`; it is not evidence of gradient descent.

## Objective and regularization boundary

The empirical objective is exact and independently recomputed. The
regularization description is intentionally conservative:

- public `penalty`, `alpha`, `C`, and `l1_ratio` values are estimator-backed;
- `C` is labeled inverse strength;
- L1, L2, and elastic-net formulas are canonical penalty-family references;
- exact internal scaling is not claimed;
- intercept penalty is `not introspected` unless no intercept penalty question
  applies;
- the canonical gradient update is never marked as the estimator's exact rule.

This prevents version-, solver-, batching-, averaging-, or schedule-specific
behavior from being presented as an established fact.

## Logistic interpolation parity correction

Before Phase 1, non-incremental logistic histories independently interpolated
probabilities and coefficients. Their endpoints were correct, but an
intermediate displayed probability did not necessarily equal the sigmoid,
Softmax, or OvR transformation of the displayed coefficient state.

For coefficient-bearing estimators, Phase 1 now interpolates parameters first
and derives every intermediate score, probability, curve/surface, and loss from
that same state. Prior baselines are represented by:

- binary log-odds for sigmoid;
- centered log priors for Softmax;
- classwise prior logits for normalized OvR.

`metadata.source_detail.interpolation_target` records `parameters`.
Probability-only estimators retain the old fallback and record
`probabilities`. Both remain labeled synthetic interpolation, not optimizer
training.

The correction keeps the same K semantic states, N retained states, Plotly
frame count, transition settings, controls, and fluid playback behavior.

## Exact fitted endpoint for reconstructed replays

Incremental replay uses a clone configured for observable one-step updates.
Those effective settings cannot reproduce every detail of the estimator's
original `fit()` execution, so a replay may approach the supplied model without
ending at identical parameters. That behavior is informative provenance, but
it is not the best endpoint for a lesson whose equations should culminate in
the model the student actually supplied.

Phase 1 therefore reserves the final semantic state for the exact supplied
estimator in both linear and logistic replay histories. The contract is
explicit rather than retrospective:

- earlier states have origin `replayed`;
- the last state has origin `fitted_estimator` and slider label `fitted`;
- the subtitle says `Reconstructed replay + fitted endpoint`;
- `source_detail.endpoint_policy` is `supplied_fitted_estimator`;
- coefficients, geometry, probabilities, raw empirical loss, and metrics are
  recomputed from the supplied model at that endpoint;
- `final_state_matches_estimator` verifies the result numerically.

The endpoint is one of the requested K semantic states, not an extra hidden
step. Temporal decimation retains it, and the hybrid renderer interpolates
smoothly from the preceding replay state without calling that visual interval
an optimizer update. Consequently the evolving LaTeX equation finishes at the
same coefficients used by the fixed fitted-model panel.

## Visual behavior

Academic binary 1D figures add a dashed horizontal threshold reference. The
intersection with the moving sigmoid makes the decision boundary discoverable.
Academic binary 2D figures add a probability-surface contour at the configured
threshold. Multiclass focus hides competing probability surfaces without
changing their data.

The fitted-model reference panel appears below the slider and uses the existing
dark visual language. It does not modify parameter matrices, plots, legends,
or controls. In the one-dimensional hybrid view, the evolving equation uses
the shared reserved math band described above; the panel does not own or move
that equation.

When `show_loss=True`, synthetic interpolation now displays its empirical path
evaluation instead of silently suppressing the requested curve. Linear paths
use MSE; logistic paths use empirical log-loss. Visible labels use
`Interpolation MSE` or `Interpolation log-loss`, and the contract records
`optimizer_loss=False`; these quantities are never described as
recovered solver loss. EMA remains available for reconstructed replay, but is
not applied to a synthetic path because that path is already smooth and must
end at the exact fitted-model value. Small expanded nD linear layouts (up to 10
variables) use a 640-pixel base height, a wider curve column, and a larger
loss-axis domain; dense matrix layouts retain their established size.

The underlying coefficient path is explicit:

\[
\theta(\alpha)=(1-\alpha)\theta_{base}+\alpha\theta_{fitted},
\qquad 0\leq\alpha\leq1.
\]

The intercept follows the same rule, and every displayed prediction and
empirical metric is recomputed from the resulting state. This is a pedagogical
model-space path used when an estimator does not expose its original fit
states; it is not called gradient descent.

## Verification

Dedicated Phase-1 tests cover:

- linear contribution sums, objective values, feature names, and prediction
  equality;
- affine StandardScaler conversion in original space;
- non-affine polynomial feature mathematics in transformed space;
- binary string-label order, custom threshold, indexed output, and probability
  equality;
- multinomial Softmax and normalized-OvR probability equality;
- selected multiclass surface visibility and 1/K title state;
- parameter-to-probability synchronization in every binary and multiclass
  interpolation frame;
- estimator-backed regularization settings and conservative claims;
- interpolation loss semantics, exact fitted endpoints, and replay-only EMA;
- multiline contribution layouts that show all six moderate-dimensional terms
  and a disclosed nine-of-twenty selection without canvas clipping;
- preservation of hybrid frame count and redraw-free motion, plus separation
  of the evolving LaTeX equation from the data axes at every detail level;
- invalid Phase-1 configuration values.

The complete 110-test suite remains a required gate. Sphinx is built with
warnings treated as errors, and the end-user notebook executes 10 code cells
and nine real Plotly figures from a clean kernel with no errors or unit-test
assertions.

## Deferred work

The following remains intentionally outside Phase 1:

- a second visual theme or responsive reflow system (Phase 3);
- user-selectable contribution sorting and expansion controls beyond the
  current compact top-contribution panel;
- a complete raw-space symbolic inverse for arbitrary non-affine transforms,
  which generally does not exist;
- exact solver-private regularization normalization;
- dynamic fitted-model panel substitution on every hybrid visual subframe,
  which would reduce motion quality without adding semantic states;
- the full generated QA notebook matrix and formal student curriculum (Phase
  4). The Phase-1 showcase notebook is a focused acceptance artifact.
