import numpy as np
import plotly.io as pio
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_logistic

pio.renderers.default = "notebook"

# ============================================================
# 1) Cargar datos de prueba sintéticos (2 variables, binario)
# ============================================================
print(">>> Generando datos sintéticos para clasificación binaria (2 variables)...")
X_full, y_full = make_classification(
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Pipeline con SGDClassifier para clasificación binaria
model_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(
        loss="log_loss",  # Regresión logística
        learning_rate="constant",
        eta0=0.01,
        max_iter=1000,
        tol=1e-6,
        random_state=42
    ))
])

# Entrenar el modelo
model_scaled.fit(X_full, y_full)

# ============================================================
# 2) Visualización en Espacio Escalado
# ============================================================
print(">>> Caso 1: Visualizando en Espacio Escalado...")
fig_scaled = visualize_logistic(
    model_scaled,
    X_full,
    y_full,
    steps=60,
    show_loss=True,
    frame_duration=40,
    title="Binary Logistic Regression (2 variables) - Scaled Space",
    display_space="scaled",
)
fig_scaled.show()

# ============================================================
# 3) Visualización en Espacio Original
# ============================================================
print(">>> Caso 2: Visualizando en Espacio Original...")
fig_orig = visualize_logistic(
    model_scaled,
    X_full,
    y_full,
    steps=60,
    show_loss=True,
    frame_duration=40,
    title="Binary Logistic Regression (2 variables) - Original Space",
    display_space="original",
)
fig_orig.show()
