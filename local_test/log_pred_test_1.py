import numpy as np
import plotly.io as pio
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import explain_logistic_prediction

pio.renderers.default = "notebook"

# -------------------------
# Synthetic dataset: Binary Classification (1 Feature)
# -------------------------
print(">>> Generando datos sintéticos para clasificación binaria (1 variable)...")
X, y = make_classification(
    n_samples=200,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Punto de consulta arbitrario
x_query = X[15:16] # Por ejemplo, la observación 15

# ============================================================
# 1) Modelo DIRECTO (sin scaler)
# ============================================================
print("\n>>> Caso 1: Modelo directo (LogisticRegression sin scaler)")
model_raw = LogisticRegression(random_state=42)
model_raw.fit(X, y)

p_hat_raw = model_raw.predict_proba(x_query)[0, 1]
y_hat_raw = model_raw.predict(x_query)[0]

fig = explain_logistic_prediction(
    model_raw, X, y,
    x_query=x_query[0],
    p_hat=p_hat_raw,
    y_hat=y_hat_raw,
    title="Logistic Regression: Predicción (Raw)",
    display_space="original"
)
fig.show()

# ============================================================
# 2) Pipeline con StandardScaler + LogisticRegression
# ============================================================
print("\n>>> Caso 2: Pipeline con StandardScaler")
model_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(random_state=42))
])
model_scaled.fit(X, y)

p_hat_scaled = model_scaled.predict_proba(x_query)[0, 1]
y_hat_scaled = model_scaled.predict(x_query)[0]

# Verlo en espacio ORIGINAL
print("   - Mostrando en espacio original (display_space='original')")
fig1 = explain_logistic_prediction(
    model_scaled, X, y,
    x_query=x_query[0],
    p_hat=p_hat_scaled,
    y_hat=y_hat_scaled,
    title="Logistic Regression (Pipeline): Predicción (Original Space)",
    display_space="original"
)
fig1.show()

# Verlo en espacio ESCALADO
print("   - Mostrando en espacio escalado (display_space='scaled')")
fig2 = explain_logistic_prediction(
    model_scaled, X, y,
    x_query=x_query[0],
    p_hat=p_hat_scaled,
    y_hat=y_hat_scaled,
    title="Logistic Regression (Pipeline): Predicción (Scaled Space)",
    display_space="scaled"
)
fig2.show()
