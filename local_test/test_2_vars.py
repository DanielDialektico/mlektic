import numpy as np
from sklearn.linear_model import SGDRegressor

from mlektic import visualize_lr

np.random.seed(0)

# --- 1. Generar datos ---
n = 100
X1 = np.random.rand(n) * 10
X2 = np.random.rand(n) * 5
y = (3.0 * X1) - (1.5 * X2) + 2.0 + np.random.randn(n) * 2.0

X = np.column_stack([X1, X2])

# --- 2. Entrenar modelo ---
model_2v = SGDRegressor(
    loss="squared_error",
    max_iter=50,
    learning_rate="constant",
    eta0=0.001,
    random_state=42,
    tol=1e-3,
    shuffle=False,
)
model_2v.fit(X, y)

# --- 3. Visualizar ---
fig_2v = visualize_lr(
    model_2v,
    X,
    y,
    steps=60,
    show_loss=True,
    title="Local Test: 2 Variabes Unscaled (Plano)",
)
fig_2v.show()
