import numpy as np
import plotly.io as pio
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr

pio.renderers.default = "notebook"

# ============================================================
# Cargar datos (TODO el df, California Housing = 8 vars)
# ============================================================
data = fetch_california_housing()
X_full = data.data
y_full = data.target

# Usamos solo 50 muestras para que la visualización densa
# sea ligera, aunque el modelo esté fatal.
np.random.seed(7)
idx = np.random.choice(len(X_full), size=50, replace=False)
X = X_full[idx]
y = y_full[idx]

model_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "sgd",
            SGDRegressor(
                loss="squared_error",
                penalty=None,
                learning_rate="constant",
                eta0=0.01,  # << con escalado, LR mucho más grande
                max_iter=2000,
                tol=1e-6,
                shuffle=False,
                random_state=7,
            ),
        ),
    ]
)

model_scaled.fit(X, y)

# Por ahora solo soportamos "original" como base para 8 vars
fig_scaled_original_theta = visualize_lr(
    model_scaled,
    X,
    y,
    steps=80,
    mode="iterative",
    show_loss=True,
    title="California Housing (8 vars) — CON StandardScaler — θ en espacio original",
    display_space="original",
)
fig_scaled_original_theta.show()
