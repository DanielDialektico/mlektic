import numpy as np
from sklearn.linear_model import LogisticRegression
from mlektic.visualization.logistic.prediction import explain_logistic_prediction

# Generamos datos sintéticos 1D con 10 clases usando un simple mapeo de rangos
print(">>> Generando datos para regresión logística multiclase 1D...")
np.random.seed(42)
X = np.linspace(-10, 10, 500).reshape(-1, 1)
# Para 10 clases, vamos a asignar clases basadas en los valores de X y añadir ruido
y = np.digitize(X.ravel() + np.random.normal(0, 0.5, size=500), bins=np.linspace(-8, 8, 9))

print(">>> Clases únicas generadas:", np.unique(y))
print(">>> Entrenando LogisticRegression (multinomial)...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)

x_query = np.array([2.5])

print(">>> Creando visualización de predicción multiclase interactiva...")
fig = explain_logistic_prediction(
    model, X, y,
    x_query=x_query,
    title="Multiclass Logistic Regression 1D (K=10) - Prediction",
    display_space="original"
)

fig.show()
