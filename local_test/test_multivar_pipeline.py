import numpy as np
import plotly.io as pio
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr

pio.renderers.default = "notebook"

# ============================================================
# CASO 1: California Housing (8 vars) CON StandardScaler
#         (ESPACIO ESCALADO vs ESPACIO ORIGINAL)
# ============================================================
print(">>> CASO 1: 8 Variables (California Housing) con Pipeline")
data = fetch_california_housing()
X_full = data.data
y_full = data.target

np.random.seed(7)
idx = np.random.choice(len(X_full), size=50, replace=False)
X_8 = X_full[idx]
y_8 = y_full[idx]

model_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "sgd",
            SGDRegressor(
                loss="squared_error",
                penalty=None,
                learning_rate="constant",
                eta0=0.01,
                max_iter=2000,
                tol=1e-6,
                shuffle=False,
                random_state=7,
            ),
        ),
    ]
)

model_scaled.fit(X_8, y_8)

# -- 1A. Espacio Escalado
fig_scaled_8v = visualize_lr(
    model_scaled,
    X_8,
    y_8,
    steps=80,
    mode="iterative",
    show_loss=True,
    title="California Housing (8 vars) — CON StandardScaler — θ en espacio escalado",
    display_space="scaled",
)
fig_scaled_8v.show()


# -- 1B. Espacio Original
fig_original_8v = visualize_lr(
    model_scaled,
    X_8,
    y_8,
    steps=80,
    mode="iterative",
    show_loss=True,
    title="California Housing (8 vars) — CON StandardScaler — θ en espacio original",
    display_space="original",
)
fig_original_8v.show()


# ============================================================
# CASO 2: Datos simulados (100 variables) SIN escalado
# ============================================================
print(">>> CASO 2: 100 Variables (Simulados) sin Pipeline")
np.random.seed(7)
n = 240
d = 100

scale_x = 0.15
b_small = 0.03
noise_std = 0.05

X_100 = scale_x * np.random.normal(0, 1.0, size=(n, d))

true_w = np.zeros(d, dtype=float)
vals = np.array([2.2, -1.5, 0.7, 0.0, -0.9, 1.3, -0.2, -0.5, 0.6, -0.1], dtype=float)
k = min(d, vals.size)
true_w[:k] = vals[:k]

y_100 = (X_100 @ true_w + b_small) + np.random.normal(0, noise_std, size=n)

model_100 = SGDRegressor(
    loss="squared_error",
    penalty=None,
    alpha=0.0,
    learning_rate="constant",
    eta0=0.02,
    shuffle=False,
    max_iter=1500,
    tol=1e-6,
    random_state=7,
)
model_100.fit(X_100, y_100)

fig_100 = visualize_lr(
    model_100,
    X_100,
    y_100,
    steps=70,
    title="Simulación (100 vars) - SGDRegressor (Sin escalado)",
)
fig_100.show()


# ============================================================
# CASO 3: Datos simulados (>100 variables, ej. 150) CON escalado, espacio original
# ============================================================
print(">>> CASO 3: 150 Variables (Simulados) con Pipeline en Espacio Original")
np.random.seed(42)
n_large = 300
d_large = 150

X_large = np.random.normal(0, 5.0, size=(n_large, d_large))
true_w_large = np.zeros(d_large, dtype=float)
true_w_large[:5] = [3.5, -2.1, 1.0, 5.0, -0.5]
y_large = (X_large @ true_w_large + 1.5) + np.random.normal(0, 1.0, size=n_large)

model_large_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "sgd",
            SGDRegressor(
                loss="squared_error",
                penalty=None,
                learning_rate="constant",
                eta0=0.015,
                max_iter=1000,
                tol=1e-6,
                shuffle=False,
                random_state=42,
            ),
        ),
    ]
)
model_large_scaled.fit(X_large, y_large)

fig_large_original = visualize_lr(
    model_large_scaled,
    X_large,
    y_large,
    steps=60,
    show_loss=True,
    title="Simulación (150 vars) - Pipeline Completo (espacio original)",
    display_space="original",
)
fig_large_original.show()
