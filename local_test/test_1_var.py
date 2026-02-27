import numpy as np
import plotly.io as pio
from sklearn.linear_model import SGDRegressor

from mlektic import visualize_lr

pio.renderers.default = "notebook"

# -------------------------
# Data (más "lenta" por escala pequeña)
# -------------------------
np.random.seed(7)
n = 120

scale_x = 0.2  # << hace X pequeño => gradientes pequeños
b_small = 0.04  # << intercepto pequeño => residuales pequeños
noise_std = 0.03  # << ruido pequeño

X = scale_x * np.random.normal(0, 1.0, size=(n, 1))
y = (2.2 * X[:, 0] + b_small) + np.random.normal(0, noise_std, size=n)

# -------------------------
# Usuario entrena normal (SIN Pipeline)
# -------------------------
model = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.02,
    shuffle=False,
    max_iter=1000,
    tol=1e-6,
    random_state=7,
)

model.fit(X, y)

print("Pred (first 5):", model.predict(X[:5]))

# -------------------------
# Visualización (tu API)
# -------------------------
fig = visualize_lr(
    model,
    X,
    y,
    steps=80,
    show_loss=True,
    title="Linear Regression (Simple, 1 variable) - Slow/Smooth Data",
)
fig.show()
