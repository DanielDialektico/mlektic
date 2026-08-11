# Mlektic

Mlektic is an interactive Python library for learning the mathematics of fitted machine-learning models. It connects Scikit-learn linear and logistic estimators and PyTorch networks to Plotly views of model geometry, parameters, probabilities, training records, and individual predictions.

Its central rule is simple: an animation must say where its states came from. A sequence reconstructed over a cloned estimator is labeled as a replay. A baseline-to-model path is labeled as synthetic interpolation. Only states observed during the actual training process are called recorded.

## What Mlektic provides

- Linear regression in 1D, 2D, and high-dimensional report views.
- Binary and multiclass logistic regression in 1D, 2D, and high-dimensional views.
- Model-aware prediction explanations with mathematical substitution.
- Original/scaled coefficient views for recognized affine Scikit-learn pipelines.
- Explicit history provenance, retained checkpoint coordinates, and smoothing metadata.
- Optional academic and complete tabular derivations with estimator-verified contributions, probabilities, objectives, feature spaces, and conservative regularization semantics.
- Fluid native and hybrid Plotly animation without presenting visual subframes as optimizer updates.
- PyTorch architecture, computational graph, training, parameter, activation, forward-pass report, and prediction views.
- Complete HTML export with explicit Plotly and MathJax dependency choices.

## Installation

```bash
pip install -e .
```

PyTorch support is optional:

```bash
pip install -e ".[torch]"
```

Notebook, documentation, and maintainer environments are explicit:

```bash
pip install -e ".[notebooks]"
pip install -e ".[docs]"
pip install -e ".[dev,torch]"
```

Core dependencies are NumPy, Scikit-learn, and Plotly. Notebook-specific IPython functionality is loaded only when requested.

## Linear regression quickstart

```python
import numpy as np
from sklearn.linear_model import SGDRegressor

from mlektic import visualize_lr

X = np.linspace(-2, 2, 80).reshape(-1, 1)
y = 1.5 + 2.2 * X[:, 0]

model = SGDRegressor(max_iter=200, random_state=7).fit(X, y)

fig = visualize_lr(
    model,
    X,
    y,
    steps=100,          # K reconstructed checkpoints
    max_frames=30,      # N displayed checkpoints
    smooth="ema",
    animation_mode="auto",
)
fig.show()
```

`SGDRegressor` supports `partial_fit`, so Mlektic constructs a replay over a clone. Because replay settings cannot reproduce every private detail of the original `fit()`, the final state is the exact supplied estimator and is explicitly labeled `fitted`; earlier states remain labeled as reconstructed replay. The figure shows N/K and preserves both retained coordinates and state origins.

For a non-incremental estimator:

```python
from sklearn.linear_model import LinearRegression

closed_form_model = LinearRegression().fit(X, y)
fig = visualize_lr(closed_form_model, X, y, steps=30)
```

This path is a synthetic interpolation from a documented baseline to the fitted model. The slider uses interpolation progress rather than “training step.”

## Logistic regression quickstart

```python
from sklearn.linear_model import LogisticRegression

from mlektic import visualize_logistic

labels = np.where(X[:, 0] >= 0, "accepted", "rejected")
classifier = LogisticRegression().fit(X, labels)

fig = visualize_logistic(
    classifier,
    X,
    labels,
    steps=30,
    multiclass_link="auto",
)
fig.show()
```

Binary views connect the linear score, sigmoid probability, fitted class order, and decision. Multiclass views resolve supported Softmax or normalized one-vs-rest probability semantics and retain the estimator's `classes_` ordering.

## Academic mathematical detail

The default `detail="essential"` keeps the compact main figure and full machine-readable contract. Use `"academic"` for a fitted-model derivation or `"complete"` for objective, metrics, preprocessing, regularization, and optimizer caveats:

```python
academic_linear = visualize_lr(
    closed_form_model,
    X,
    y,
    detail="complete",
    feature_names=["study_hours"],
    sample_index=12,
)

academic_logistic = visualize_logistic(
    classifier,
    X,
    labels,
    detail="academic",
    threshold=0.65,
    sample_index=12,
)
```

The added panel is explicitly a **fitted-model derivation**. It stays fixed while the existing animation shows its replayed or interpolated states, preserving fluid hybrid motion. In one-dimensional linear regression, every detail level uses the same evolving LaTeX fitted equation in a reserved band above the axes; metric cards remain in their side column and the equation never covers the data. The machine-readable calculation is available in every detail level:

```python
contract = academic_linear.layout.meta["mlektic_math"]
contract["sample"]["contributions"]
contract["sample"]["matches_model"]
contract["objective"]
contract["regularization"]
```

For affine preprocessing, Mlektic verifies the transformation and can express coefficients in original units. For a non-affine transformation such as polynomial expansion, it shows transformed-feature mathematics and uses `get_feature_names_out` when available; it does not invent a raw-space coefficient vector.

Linear refers to linearity in the fitted coefficients, not necessarily in the original input. A pipeline with `PolynomialFeatures(2)` fits `theta_0 + theta_1*x + theta_2*x^2`, which is linear in theta but correctly appears as a parabola against the original `x`. Its separate `Interpolation MSE` is empirical evaluation along a declared baseline-to-model parameter path, not gradient-descent history.

For multiclass views, `class_focus` accepts a fitted class label or zero-based index and keeps one probability curve or surface visible. The title reports `1/K`, while the complete fitted class order remains in metadata:

```python
focused = visualize_logistic(
    multiclass_model,
    X_multiclass,
    y_multiclass,
    detail="academic",
    class_focus="setosa",
)
```

`show_objective="auto"` enables the empirical objective in academic/complete detail. `show_regularization="auto"` enables its summary in complete detail. Mlektic reports only public penalty settings (`penalty`, `alpha` or inverse-strength `C`, and `l1_ratio`) and labels unknown private normalization or intercept-penalty behavior as not introspected.

## Prediction explanations

```python
from mlektic import explain_lr_prediction, explain_logistic_prediction

linear_explanation = explain_lr_prediction(
    closed_form_model,
    X,
    y,
    x_query=[[1.25]],
)

logistic_explanation = explain_logistic_prediction(
    classifier,
    X,
    labels,
    x_query=[[0.75]],
    show_class_labels=False,
)
```

The explainers always compute the estimator output. Supplied `yhat`, `p_hat`, or `y_hat` values are verified by default:

```python
predicted = closed_form_model.predict([[1.25]])[0]
fig = explain_lr_prediction(
    closed_form_model,
    X,
    y,
    x_query=[[1.25]],
    yhat=predicted,
)
```

Use `prediction_source="provided"` only for an intentional counterfactual lesson. The figure will identify it as user-provided. Queries outside any observed per-feature training range are marked as extrapolations and remain visible in 1D/2D plots.

Logistic figures use class indices by default. In a binary explanation, the displayed sigmoid value is `p_1`, the output compares `(p_0, p_1)`, and the winning index is reported without semantic labels. Set `show_class_labels=True` to append fitted labels from `classes_` to the indexed axes, legends, and winning class. Labels and fitted order remain available in `layout.meta` in either mode. For two features, observed class targets are plotted numerically at 0 and 1, the trained model is a probability surface, and the decision boundary is the line where that surface crosses `p_1 = 0.5`.

## History contract

Tabular history payloads expose an auditable contract:

```python
history["metadata"]
# {
#   "source": "replayed" | "interpolated",
#   "requested_mode": "auto",
#   "resolved_mode": "iterative",
#   "requested_steps": 100,
#   "training_total_steps": 200,
#   "captured_steps": 100,
#   "displayed_steps": 30,
#   "step_indices": array([...]),
#   "displayed_step_indices": array([...]),
#   "final_state_matches_estimator": False,
#   "display_space": "original",
#   "smoothing": {"method": "ema", "beta": 0.85},
#   "warnings": [...],
# }
```

Time vocabulary:

- **T** — actual/reported training updates, when known.
- **K** — semantic checkpoints recorded or constructed before display sampling.
- **N** — retained checkpoints displayed after sampling.
- **q** — perceptual intervals inserted between displayed checkpoints.
- **F** — Plotly visual frames; these are not additional optimizer updates.

Read [History provenance and time semantics](codeasdoc/history_semantics.rst) before interpreting convergence from an animation.

## Raw and displayed loss

```python
history["loss_raw"]
history["loss_display"]
history["loss_hist"]       # backward-compatible alias of loss_display
```

EMA never overwrites the empirical series. It is applied to reconstructed replays, where successive updates may be visually noisy. Synthetic interpolation already defines a smooth mathematical path, so its visible empirical MSE or log-loss remains raw and reaches the exact fitted endpoint. Curves and metric cards state `Replay`, `Interpolation`, or `EMA` semantics explicitly; metadata records the quantity, role, smoothing decision, and that it is not an introspected private optimizer loss.

## Animation controls

Important linear controls include:

- `animation_mode="auto" | "native" | "hybrid"`
- `frame_duration`
- `transition_duration`
- `fps`
- `interpolation_frames`
- `max_frames`
- `frame_step`

`auto` uses hybrid trace-only motion for one-dimensional linear regression and native animation elsewhere. Hybrid subframes improve continuity; semantic labels advance only at retained checkpoints. The symbolic definition is fixed and the numerical fitted equation is another synchronized LaTeX trace in a reserved math band, so playback does not require layout redraws.

Important shared history controls include:

- `mode="auto" | "iterative" | "final_interp"`
- `baseline="mean" | "zeros"` for linear interpolation
- `baseline="prior" | "uniform"` for logistic interpolation
- `display_space="original" | "scaled"`
- `smooth=None | "ema"`
- built-in metric sequences or custom callable mappings
- `show_history_context=True | False` to show or hide only the provenance/N-K
  subtitle; the slider and `layout.meta` always retain the context

Unsupported values fail early with an English error. Explicit iterative mode requires an estimator with `partial_fit`.

## PyTorch neural networks

```python
import torch
from torch import nn

from mlektic import (
    TorchTrainingRecorder,
    build_nn_math_report,
    explain_nn_prediction,
    visualize_nn_architecture,
    visualize_nn_graph,
    visualize_nn_training,
    visualize_nn_weights,
)

network = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

architecture = visualize_nn_architecture(network, input_shape=(4,))
graph = visualize_nn_graph(network, input_shape=(4,))
weights = visualize_nn_weights(network)
```

For genuine training history, create a recorder and call it from the training loop at a consistent point:

```python
recorder = TorchTrainingRecorder(network, record_every=5)

for epoch in range(100):
    optimizer.zero_grad()
    predictions = network(inputs)
    loss = criterion(predictions, targets)
    loss.backward()
    optimizer.step()

    recorder.record(step=epoch + 1, loss=float(loss.detach()))

training_figure = visualize_nn_training(recorder)
```

The mathematical report and prediction explainer are separate APIs so large derivations do not overload one interactive plot.

## HTML export

```python
from mlektic import export_figure

path = export_figure(
    fig,
    "linear-lesson.html",
    include_plotly="inline",
    include_mathjax="cdn",
    responsive=False,
    auto_play=False,
)
```

The default inlines Plotly but loads MathJax from a CDN. The Plotly runtime is offline-capable; equation rendering still requires network access. Passing `include_mathjax=False` preserves the LaTeX source but does not guarantee rendered equations. Fully self-contained MathJax is not currently promised.

## Visual themes, formats, and sizes

Current classic width, height, styling, and motion remain the defaults. Phase 3
adds independent, opt-in visual axes:

```python
academic_lesson = visualize_lr(
    model,
    X,
    y,
    theme="academic",
    format="lesson",
    density="academic",
    size="notebook",
    responsive=True,
)
```

- `theme` controls color, typography, and line/marker styling: `classic`,
  `academic`, `classroom`, `compact`, or `accessible`.
- `format` controls composition: the existing `dashboard`, staged `lesson`,
  space-conscious `compact`, or static final-state `report`.
- `density` is a compatible alias for tabular mathematical `detail`.
- `size` accepts `default`, `compact`, `notebook`, `wide`, or `classroom`;
  explicit `width` and `height` override the preset.
- `responsive=True` scales the selected composition. Choose another `format`
  when structural reflow is needed.
- `reduced_motion=True` provides the exact final displayed state without
  playback controls.

`dashboard`, `compact`, and `lesson` retain every selected animation frame.
The lesson stages only change trace visibility; Play/Pause and fluid motion
remain available. Only `report` and `reduced_motion` intentionally create a
static view. The `accessible` theme uses marker symbols and line dashes in
addition to color.

Every resolved choice and token is inspectable in
`figure.layout.meta["mlektic_visual"]`. `export_figure()` inherits responsive
behavior from this metadata when its `responsive` argument is omitted. See the
[visual design guide](codeasdoc/visual_design.rst) and the
[Phase 3 implementation record](mejoras/10_phase_3_implementation_record.md).

## Project architecture

```text
src/mlektic/
  api/              public orchestration and export
  services/         history service facades
  adapters/         Scikit-learn estimator/pipeline normalization
  domain/           validated config and payload contracts
  history/          replay, interpolation, metrics, sampling, provenance
  visualization/    linear, logistic, neural Plotly builders
  neural/           recorder, introspection, reports
  utils/            numerical and probability helpers
```

See [Architecture](codeasdoc/architecture.rst), [API reference](codeasdoc/api_reference.rst), [compatibility](codeasdoc/compatibility.rst), and [limitations](codeasdoc/limitations.rst).

## Testing

```bash
pytest
python -m ruff check src tests scripts codeasdoc
python scripts/validate_notebook_policy.py
```

Canonical notebooks are separated by audience:

- [`notebooks/learn`](notebooks/learn) contains focused student lessons;
- [`notebooks/qa`](notebooks/qa) contains 98 stable human visual-QA cases across
  model families, dimensionalities, data regimes, parameters, hyperparameters,
  provenance, motion, predictions, themes, formats, density, and sizes;
- [`notebooks/archive`](notebooks/archive) preserves earlier phase notebooks and
  inventories the large local exploratory notebooks without deleting them.

Regenerate and validate the canonical suite with:

```bash
python scripts/generate_notebooks.py
python scripts/execute_notebooks.py --group smoke
```

Every new or materially changed public documentation page must add a new,
separately executable visual cell with a stable case ID to the corresponding QA
notebook. This contract is enforced by CI; see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation

Build Sphinx locally:

```bash
sphinx-build -W -b html codeasdoc codeasdoc/_build/html
```

Canonical new documentation, docstrings, error messages, tests, and planning records are English-first.

## Contributing

Before adding a visual feature:

1. define whether its states are recorded, replayed, or interpolated;
2. define the exact mathematics and estimator assumptions;
3. preserve raw values and source coordinates;
4. add invariant tests before screenshot tests;
5. keep the classic default stable unless a versioned change is approved;
6. document unsupported cases explicitly.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
