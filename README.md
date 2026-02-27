# Mlektic

Mlektic is a Python library designed to visually demonstrate how machine learning models evolve during training. 
It provides interactive, mathematical, and highly visual plots compatible with `scikit-learn`.

## Installation

Mlektic uses `uv` for dependency management. To set up the project locally:

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository and install dependencies:
```bash
git clone https://github.com/DanielDialektico/mlektic.git
cd mlektic
uv venv
uv sync
```

## Quickstart

```python
import numpy as np
from sklearn.linear_model import SGDRegressor
from mlektic.core import fit_history, build_lr_figure

# Generate some dummy data
X = np.random.rand(100, 1) * 10
y = 2.5 * X.ravel() + np.random.randn(100) * 2

# Create your model
model = SGDRegressor(max_iter=100)

# Capture training history
history = fit_history(model, X, y, steps=20)

# Build an animated Plotly figure
fig = build_lr_figure(X, y, history=history)
fig.show()
```

## Contributing

We welcome contributions! Please make sure to install development dependencies and run tests/linters before submitting a PR.

```bash
# Run tests
uv run pytest

# Check code formatting and linting
uv run ruff check .
uv run ruff format --check .
```
