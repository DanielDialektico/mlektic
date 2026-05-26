import numpy as np
import plotly.io as pio
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr

pio.renderers.default = "notebook"

# ============================================================
# 1) Cargar datos (2 variables)
# ============================================================
data = fetch_california_housing()
# Usamos MedInc y HouseAge
X_full = data.data[:, [0, 1]]
y_full = data.target

np.random.seed(7)
idx = np.random.choice(len(X_full), size=500, replace=False)
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
                eta0=0.001,  # con escalado puedes subir LR bastante
                max_iter=2000,
                tol=1e-6,
                shuffle=False,
                random_state=7,
            ),
        ),
    ]
)

model_scaled.fit(X, y)

# Visualizar Theta en ESPACIO ESCALADO
fig_scaled_theta = visualize_lr(
    model_scaled,
    X,
    y,
    steps=70,
    show_loss=True,
    title="California Housing (2 vars) — CON StandardScaler — θ en espacio escalado",
    display_space="scaled",
)
fig_scaled_theta.show()

# Visualizar Theta en ESPACIO ORIGINAL
fig_orig_theta = visualize_lr(
    model_scaled,
    X,
    y,
    steps=70,
    show_loss=True,
    title="California Housing (2 vars) — CON StandardScaler — θ en espacio original",
    display_space="original",
)
fig_orig_theta.show()
