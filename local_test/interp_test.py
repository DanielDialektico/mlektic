import numpy as np
import plotly.io as pio
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression

from mlektic import visualize_lr, visualize_logistic

pio.renderers.default = "notebook"

# ============================================================
# 1. REGRESIÓN LINEAL (Mínimos Cuadrados Ordinarios)
# ============================================================
print(">>> Ejemplo 1: Regresión Lineal (OLS) - Interpolación Artificial")
X_lin, y_lin = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

model_lin = LinearRegression()
model_lin.fit(X_lin, y_lin)

# Como LinearRegression no es iterativo, mode="auto" usará InterpolationCapture.
fig_lin = visualize_lr(
    model_lin,
    X_lin,
    y_lin,
    steps=60,
    show_loss=True,
    title="LinearRegression (OLS) - Evolución Interpolada",
    frame_duration=30,
)
fig_lin.show()


# ============================================================
# 2. REGRESIÓN LOGÍSTICA (L-BFGS / Sin historial)
# ============================================================
print(">>> Ejemplo 2: Regresión Logística (L-BFGS) - Interpolación Artificial")
X_log, y_log = make_classification(
    n_samples=200, 
    n_features=1, 
    n_informative=1, 
    n_redundant=0, 
    n_clusters_per_class=1, 
    class_sep=1.0, 
    random_state=42
)

# LogisticRegression de Sklearn por defecto usa el solver lbfgs
# y nuestro motor sabe que no puede extraer historial iterativo de ahí de forma nativa.
model_log = LogisticRegression()
model_log.fit(X_log, y_log)

fig_log = visualize_logistic(
    model_log,
    X_log,
    y_log,
    steps=80,
    show_loss=True,
    title="LogisticRegression (L-BFGS) - Evolución Interpolada",
    frame_duration=30,
)
fig_log.show()
