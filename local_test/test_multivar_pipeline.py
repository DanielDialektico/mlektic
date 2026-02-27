import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr

np.random.seed(7)

# --- 1. Generar datos multivariables ---
n = 150
d = 8
X = np.random.randn(n, d) * 10
# Coeficientes verdaderos
true_w = np.array([1.5, -2.0, 0.0, 0.5, 3.2, -1.1, 0.0, 0.2])
y = X @ true_w + 5.0 + np.random.randn(n) * 5.0

# --- 2. Entrenar modelo con PIPELINE ---
model_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "sgd",
            SGDRegressor(
                loss="squared_error",
                penalty=None,
                learning_rate="constant",
                eta0=0.01,  # Con escalado, LR puede ser mayor
                max_iter=50,
                tol=1e-6,
                shuffle=False,
                random_state=7,
            ),
        ),
    ]
)

model_scaled.fit(X, y)

# --- 3. Visualizar Theta en ESPACIO ORIGINAL (usando StandardScaler) ---
fig_scaled_original_theta = visualize_lr(
    model_scaled,
    X,
    y,
    steps=80,
    mode="iterative",
    show_loss=True,
    title="Local Test: 8 vars — StandardScaler — θ original",
    display_space="original",
)
fig_scaled_original_theta.show()

# --- 4. Visualizar Theta en ESPACIO ESCALADO ---
fig_scaled_theta = visualize_lr(
    model_scaled,
    X,
    y,
    steps=80,
    mode="iterative",
    show_loss=True,
    title="Local Test: 8 vars — StandardScaler — θ escalado",
    display_space="scaled",
)
fig_scaled_theta.show()
