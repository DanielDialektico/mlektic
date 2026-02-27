import numpy as np
import plotly.io as pio
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_lr

pio.renderers.default = "notebook"

# ============================================================
# CASO 1: California Housing (1 variable) SIN escalado
# ============================================================
print(">>> Caso 1: Datos Reales sin escalado (California Housing)")
data = fetch_california_housing()
X_full = data.data[:, [0]]  # usamos solo una variable: MedInc (ingreso medio)
y_full = data.target  # precio medio de vivienda

# Subsample pequeño para animación fluida
np.random.seed(7)
idx = np.random.choice(len(X_full), size=300, replace=False)
X_cal = X_full[idx]
y_cal = y_full[idx]

model_plain = SGDRegressor(
    loss="squared_error",
    learning_rate="constant",
    eta0=0.0005,  # pequeño porque X está en escala natural
    max_iter=2000,
    tol=1e-6,
    shuffle=False,
    random_state=7,
)
model_plain.fit(X_cal, y_cal)

fig_plain = visualize_lr(
    model_plain,
    X_cal,
    y_cal,
    steps=70,
    show_loss=True,
    title="California Housing — SGDRegressor (SIN escalado)",
    display_space="original",  # aquí scaled == original, no hay scaler
)
fig_plain.show()


# ============================================================
# CASO 2: California Housing CON escalado (visualización escalada)
# ============================================================
print(">>> Caso 2: Datos Reales CON escalado (Visualización Escalada)")
model_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "sgd",
            SGDRegressor(
                loss="squared_error",
                learning_rate="constant",
                eta0=0.001,  # ahora podemos usar lr más grande gracias al escalado
                max_iter=2000,
                tol=1e-6,
                shuffle=False,
                random_state=7,
            ),
        ),
    ]
)
model_scaled.fit(X_cal, y_cal)

fig_scaled_theta = visualize_lr(
    model_scaled,
    X_cal,
    y_cal,
    steps=70,
    show_loss=True,
    title="California Housing — SGDRegressor (CON StandardScaler) — θ en espacio escalado",
    display_space="scaled",
)
fig_scaled_theta.show()


# ============================================================
# CASO 3: California Housing CON escalado (visualización en espacio original)
# ============================================================
print(">>> Caso 3: Datos Reales CON escalado (Visualización Original)")
fig_original_theta = visualize_lr(
    model_scaled,
    X_cal,
    y_cal,
    steps=70,
    show_loss=True,
    title="California Housing — SGDRegressor (CON StandardScaler) — θ en espacio original",
    display_space="original",
)
fig_original_theta.show()


# ============================================================
# CASO 4: Datos manuales (sin pipeline)
# ============================================================
print(">>> Caso 4: Datos simulados (lentos/suaves)")
np.random.seed(7)
n = 120
scale_x = 0.2  # << hace X pequeño => gradientes pequeños
b_small = 0.04  # << intercepto pequeño => residuales pequeños
noise_std = 0.03  # << ruido pequeño

X_sim = scale_x * np.random.normal(0, 1.0, size=(n, 1))
y_sim = (2.2 * X_sim[:, 0] + b_small) + np.random.normal(0, noise_std, size=n)

model_sim = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.02,
    shuffle=False,
    max_iter=1000,
    tol=1e-6,
    random_state=7,
)
model_sim.fit(X_sim, y_sim)

print("Pred (first 5):", model_sim.predict(X_sim[:5]))

fig_sim = visualize_lr(
    model_sim,
    X_sim,
    y_sim,
    steps=80,
    show_loss=False,
    title="Linear Regression (Simple, 1 variable) - Slow/Smooth Data",
)
fig_sim.show()


# ============================================================
# CASO 5: Variación Extra: Sin suavidad en loss y baseline alternativo
# ============================================================
print(">>> Caso 5: Variación Extra (smooth='none', baseline='zero')")
fig_sim_extra = visualize_lr(
    model_sim,
    X_sim,
    y_sim,
    steps=50,
    show_loss=True,
    smooth="none",
    baseline="zeros",  # Fija la referencia máxima del loss a 0 en la vista original
    title="Variación Extra: smooth='none' y baseline='zeros'",
)
fig_sim_extra.show()
