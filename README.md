# Mlektic

Mlektic is an interactive Python library for learning the mathematics of fitted machine-learning models. It connects Scikit-learn linear and logistic estimators and PyTorch networks to Plotly views of model geometry, parameters, probabilities, training records, and individual predictions.

Its central rule is simple: an animation must say where its states came from. A sequence reconstructed over a cloned estimator is labeled as a replay. A baseline-to-model path is labeled as synthetic interpolation. Only states observed during the actual training process are called recorded.

## What Mlektic provides

- Linear regression in 1D, 2D, and high-dimensional report views.
- Binary and multiclass logistic regression in 1D, 2D, and high-dimensional views.
- Model-aware prediction explanations with mathematical substitution.
- Original/scaled coefficient views for recognized affine Scikit-learn pipelines.
- Explicit history provenance, retained checkpoint coordinates, and smoothing metadata.
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

`SGDRegressor` supports `partial_fit`, so Mlektic constructs a replay over a clone. The figure identifies the replay, shows N/K, preserves retained source indices, and reports whether the final replay parameters match the supplied fitted estimator.

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

EMA never overwrites the empirical series. A visible Loss/Log-loss metric and the loss curve use the same display values, while metadata records the method and beta.

## Animation controls

Important linear controls include:

- `animation_mode="auto" | "native" | "hybrid"`
- `frame_duration`
- `transition_duration`
- `fps`
- `interpolation_frames`
- `max_frames`
- `frame_step`

`auto` uses hybrid trace-only motion for one-dimensional linear regression and native animation elsewhere. Hybrid subframes improve continuity; semantic labels advance only at retained checkpoints.

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

Current classic width, height, styling, and motion remain the defaults. Optional academic, compact, classroom, accessible, and responsive/reflow formats are planned in [`mejoras`](mejoras/README.md), not exposed as current APIs.

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

See [Architecture](codeasdoc/architecture.rst), [API reference](codeasdoc/api_reference.rst), and [advanced usage](codeasdoc/advanced.rst).

## Testing

```bash
pytest
ruff check src tests codeasdoc
```

The current exploratory notebooks remain at repository root:

- `test_interpt.ipynb`
- `test_linreg.ipynb`
- `test_logreg.ipynb`
- `test_ann.ipynb`

The improvement plan proposes separating them into generated QA matrices and focused student learning notebooks without deleting their useful coverage.

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
