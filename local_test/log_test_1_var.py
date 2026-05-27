import webbrowser

import numpy as np
import plotly.io as pio
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic import visualize_logistic

pio.renderers.default = "notebook"

# Generación de conjunto de datos propio para clasificación binaria
# Usaremos make_classification para obtener un dataset simple sin requerir Pandas
print(">>> Generando datos sintéticos para clasificación...")
X_full, y_full = make_classification(
    n_samples=300,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# ============================================================
# CASO 1: Dataset personalizado (1 variable) SIN escalado
# ============================================================
print(">>> Caso 1: Datos propios sin escalado")

model_plain = SGDClassifier(
    loss="log_loss",  # equivalente a regresión logística
    learning_rate="constant",
    eta0=0.01,
    max_iter=1000,
    tol=1e-6,
    shuffle=False,
    random_state=42,
)
model_plain.fit(X_full, y_full)

fig_plain = visualize_logistic(
    model_plain,
    X_full,
    y_full,
    steps=70,
    show_loss=True,
    frame_duration=10,
    title="Clasificación Custom — SGDClassifier (SIN escalado)",
    display_space="original",
)
fig_plain.show()

# Nota: Puedes exportar a HTML si lo deseas usando fig_plain.write_html("output.html")


# ============================================================
# CASO 2: Dataset personalizado CON escalado (visualización escalada)
# ============================================================
print(">>> Caso 2: Datos propios CON escalado (Visualización Escalada)")
model_scaled = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "sgd",
            SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=0.05,
                max_iter=1000,
                tol=1e-6,
                shuffle=False,
                random_state=42,
            ),
        ),
    ]
)
model_scaled.fit(X_full, y_full)

fig_scaled_theta = visualize_logistic(
    model_scaled,
    X_full,
    y_full,
    steps=70,
    show_loss=True,
    frame_duration=10,
    title="Clasificación Custom — SGDClassifier (CON StandardScaler) — θ en espacio escalado",
    display_space="scaled",
)
fig_scaled_theta.show()


# ============================================================
# CASO 3: Dataset personalizado CON escalado (visualización en espacio original)
# ============================================================
print(">>> Caso 3: Datos propios CON escalado (Visualización Original)")
fig_original_theta = visualize_logistic(
    model_scaled,
    X_full,
    y_full,
    steps=70,
    show_loss=True,
    frame_duration=10,
    title="Clasificación Custom — SGDClassifier (CON StandardScaler) — θ en espacio original",
    display_space="original",
)
fig_original_theta.show()


# ============================================================
# CASO 4: Datos manuales (sin pipeline)
# ============================================================
print(">>> Caso 4: Datos manuales (sin pipeline) - Sin gráfica de Loss")
np.random.seed(42)
n = 150
# Construimos dos grupos (blobs) manualmente para las clases 0 y 1
X_class0 = np.random.normal(-2, 0.8, size=(n // 2, 1))
y_class0 = np.zeros(n // 2)

X_class1 = np.random.normal(2, 0.8, size=(n // 2, 1))
y_class1 = np.ones(n // 2)

X_manual = np.vstack([X_class0, X_class1])
y_manual = np.concatenate([y_class0, y_class1])

# Mezclar (shuffle)
idx = np.random.permutation(n)
X_manual = X_manual[idx]
y_manual = y_manual[idx]

model_manual = SGDClassifier(
    loss="log_loss",
    penalty=None,
    learning_rate="constant",
    eta0=0.02,
    shuffle=False,
    max_iter=500,
    tol=1e-6,
    random_state=42,
)
model_manual.fit(X_manual, y_manual)

fig_manual = visualize_logistic(
    model_manual,
    X_manual,
    y_manual,
    steps=60,
    show_loss=False,  # CASO sin la gráfica de métricas/loss
    frame_duration=10,
    title="Regresión Logística Manual (1 variable) - Sin Loss Chart",
)
fig_manual.show()


# ============================================================
# CASO 5: Datos con alta superposición (Muestra variación de Accuracy/F1)
# ============================================================
print(">>> Caso 5: Datos con alta superposición para notar variación de métricas")
X_overlap, y_overlap = make_classification(
    n_samples=400,
    n_features=1,
    n_informative=1,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.4,  # Baja separación para forzar errores en la frontera
    random_state=13
)

model_overlap = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.005,  # LR pequeño para que el modelo tome varios pasos explorando
    max_iter=1000,
    tol=1e-6,
    shuffle=False,
    random_state=13,
)
model_overlap.fit(X_overlap, y_overlap)

fig_overlap = visualize_logistic(
    model_overlap,
    X_overlap,
    y_overlap,
    steps=90,
    show_loss=True,
    frame_duration=20,
    title="Clasificación Custom — Alta Superposición (Variación de Métricas)",
    display_space="original",
)
fig_overlap.show()

