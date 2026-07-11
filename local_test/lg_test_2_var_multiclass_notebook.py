import numpy as np
import plotly.io as pio
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from mlektic import visualize_logistic

# Opcional: configurar para que funcione mejor en ciertos notebooks si tienes problemas de visualización
# pio.renderers.default = "notebook" 

# =========================================================
# 1. PRUEBA CON DATOS SINTÉTICOS
# =========================================================
# Cambia este valor para probar con diferente número de clases (K >= 2)
NUM_CLASSES = 3

X_syn, y_syn = make_blobs(
    n_samples=300, 
    n_features=2, 
    centers=NUM_CLASSES, 
    cluster_std=1.5,
    random_state=42
)

model_syn = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.01,
    max_iter=300,
    random_state=42
)
model_syn.fit(X_syn, y_syn)

# Gráfica 1: Datos Sintéticos CON Log-loss
fig_syn_loss = visualize_logistic(
    model_syn, X_syn, y_syn,
    steps=60,
    show_loss=True,
    frame_duration=80,
    title="Multiclase 2D (Datos Sintéticos) - CON Curva Loss",
    dec=3
)
fig_syn_loss.show()

# Gráfica 2: Datos Sintéticos SIN Log-loss
fig_syn_no_loss = visualize_logistic(
    model_syn, X_syn, y_syn,
    steps=60,
    show_loss=False,
    frame_duration=80,
    title="Multiclase 2D (Datos Sintéticos) - SIN Curva Loss",
    dec=3
)
fig_syn_no_loss.show()


# =========================================================
# 2. PRUEBA CON DATOS REALES (IRIS)
# =========================================================
iris = load_iris()
# Tomamos solo las primeras 2 variables (Largo y Ancho del Sépalo)
X_real = iris.data[:, :2]  
y_real = iris.target

scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real)

model_real = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.05,
    max_iter=500,
    random_state=42
)
model_real.fit(X_real_scaled, y_real)

# Gráfica 3: Datos Reales CON Log-loss
fig_real_loss = visualize_logistic(
    model_real, X_real_scaled, y_real,
    steps=60,
    show_loss=True,
    frame_duration=80,
    title="Multiclase 2D (Dataset Iris) - CON Curva Loss",
    dec=3
)
fig_real_loss.show()

# Gráfica 4: Datos Reales SIN Log-loss
fig_real_no_loss = visualize_logistic(
    model_real, X_real_scaled, y_real,
    steps=60,
    show_loss=False,
    frame_duration=80,
    title="Multiclase 2D (Dataset Iris) - SIN Curva Loss",
    dec=3
)
fig_real_no_loss.show()
