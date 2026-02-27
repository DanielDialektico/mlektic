import numpy as np
from sklearn.linear_model import SGDRegressor

from mlektic import visualize_lr

np.random.seed(42)

# --- 1. Generar datos (ruidosos para dificultar) ---
n_samples = 150
X = np.sort(np.random.rand(n_samples)) * 10
y = 2.5 * X + 1.0 + np.random.randn(n_samples) * 3

X = X.reshape(-1, 1)

# --- 2. Entrenar modelo (muy sub-entrenado para ver evolución) ---
model_1v = SGDRegressor(
    loss="squared_error",
    max_iter=50,
    learning_rate="constant",
    eta0=0.005,
    random_state=42,
    tol=None,
    shuffle=False,
)
model_1v.fit(X, y.ravel())

# --- 3. Visualizar ---
fig_1v = visualize_lr(model_1v, X, y, steps=60, show_loss=True, title="Local Test: 1 Var Unscaled")
fig_1v.show()

# Para el test con baseline "mean"
fig_1v_base = visualize_lr(model_1v, X, y, steps=30, baseline="mean", title="Local Test: 1 Var (baseline='mean')")
fig_1v_base.show()
