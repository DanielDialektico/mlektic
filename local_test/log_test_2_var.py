import numpy as np
import plotly.io as pio
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mlektic import visualize_logistic

pio.renderers.default = "notebook"

# ============================================================
# CASO 1: Multiclase (K=10, d=1) SIN gráfica de Loss
# ============================================================
print(">>> Caso 1: Multiclase K=10, d=1 (show_loss=False)")
X1, y1 = make_blobs(
    n_samples=400,
    centers=10,
    n_features=1,
    cluster_std=1.2,
    random_state=42
)

# Pipeline con SGDClassifier para clasificación multiclase
model1 = Pipeline([
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
model1.fit(X1, y1)

fig1 = visualize_logistic(
    model1,
    X1,
    y1,
    steps=60,
    show_loss=False,  # <--- Como en la primera imagen
    frame_duration=30,
    title="Multiclass Logistic Regression (K=10, d=1)",
    display_space="original",
)
fig1.show()


# ============================================================
# CASO 2: Multiclase (K=7, d=1) CON gráfica de Loss
# ============================================================
print("\n>>> Caso 2: Multiclase K=7, d=1 (show_loss=True)")
X2, y2 = make_blobs(
    n_samples=300,
    centers=7,
    n_features=1,
    cluster_std=1.0,
    random_state=123
)

model2 = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        max_iter=1000,
        tol=1e-6,
        random_state=123
    ))
])
model2.fit(X2, y2)

fig2 = visualize_logistic(
    model2,
    X2,
    y2,
    steps=60,
    show_loss=True,  # <--- Como en la segunda imagen
    frame_duration=30,
    title="Multiclass Logistic Regression (K=7, d=1)",
    display_space="original",
)
fig2.show()

# ============================================================
# CASO 3: Multiclase (K=20, d=20)
# ============================================================
print("\n>>> Caso 3: Multiclase K=20, d=20")
X3, y3 = make_blobs(
    n_samples=500,
    centers=20,
    n_features=20,
    cluster_std=2.0,
    random_state=42
)

model3 = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.01,
        max_iter=1000,
        tol=1e-6,
        random_state=42
    ))
])
model3.fit(X3, y3)

fig3 = visualize_logistic(
    model3,
    X3,
    y3,
    steps=60,
    show_loss=True,
    frame_duration=30,
    title="Multiclass Logistic Regression (K=20, d=20)",
    display_space="original",
)
fig3.show()
