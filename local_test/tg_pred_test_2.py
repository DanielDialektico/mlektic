import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic.api.linear import explain_lr_prediction

# -------------------------
# Real data: Diabetes (10 features)
# -------------------------
data = load_diabetes()
X = data.data.astype(float)   # shape (n, 10)
y = data.target.astype(float) # shape (n,)

d = X.shape[1]
print("d =", d)

# ------------------------------------------------------------
# Punto de consulta elegido por el usuario (como sklearn.predict)
# Aquí uso: un ejemplo "realista" = tomar una fila del dataset
# (puedes reemplazar esto por tu propio vector de 10 valores)
# ------------------------------------------------------------
i = 0
x_query = X[i:i+1]  # shape (1, 10)

# ============================================================
# 1) MODELO DIRECTO (sin escalado)
# ============================================================
model_raw = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.01,
    shuffle=False,
    max_iter=40000,
    tol=1e-8,
    random_state=7,
)
model_raw.fit(X, y)

yhat_raw = float(model_raw.predict(x_query)[0])

fig = explain_lr_prediction(
    model_raw,
    X, y,
    x_query=x_query,
    yhat=yhat_raw,
    display_space="original",
)
fig.show()

# ============================================================
# 2) PIPELINE: StandardScaler + SGDRegressor
# ============================================================
model_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.02,
        shuffle=False,
        max_iter=40000,
        tol=1e-8,
        random_state=7,
    ))
])

model_scaled.fit(X, y)

yhat_scaled = float(model_scaled.predict(x_query)[0])

# Mostrar en espacio ORIGINAL
fig = explain_lr_prediction(
    model_scaled,
    X, y,
    x_query=x_query,
    yhat=yhat_scaled,
    title="Diabetes (Pipeline + StandardScaler): 10 variables → target (original space)",
    display_space="original",
)
fig.show()

# Mostrar en espacio ESCALADO
fig = explain_lr_prediction(
    model_scaled,
    X, y,
    x_query=x_query,
    yhat=yhat_scaled,
    title="Diabetes (Pipeline + StandardScaler): 10 variables → target (scaled space)",
    display_space="scaled",
)
fig.show()


# -------------------------
# Synthetic data: 15 features
# -------------------------
np.random.seed(7)

n = 300        # number of samples
d = 15         # number of features

# Generate input matrix
X = np.random.normal(0, 1.0, size=(n, d))

# True underlying parameters (only first few nonzero, rest small noise)
true_w = np.zeros(d)
true_w[:5] = [2.5, -1.7, 0.9, 0.0, -1.2]   # meaningful coefficients
true_w[5:] = np.random.uniform(-0.2, 0.2, size=d-5)

true_b = 0.7

# Generate target with noise
y = X @ true_w + true_b + np.random.normal(0, 0.8, size=n)

print("d =", X.shape[1])

# ------------------------------------------------------------
# User-chosen query point (like sklearn.predict)
# Here we simply take one realistic sample from X
# ------------------------------------------------------------
i = 0
x_query = X[i:i+1]   # shape (1, 15)

# ============================================================
# Model: SGDRegressor + StandardScaler (optional but realistic)
# ============================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.02,
        shuffle=False,
        max_iter=40000,
        tol=1e-8,
        random_state=7,
    ))
])

model.fit(X, y)

# Standard sklearn prediction
yhat = float(model.predict(x_query)[0])

# ============================================================
# Visualization with your explainer
# ============================================================
fig = explain_lr_prediction(
    trained_estimator=model,
    X_train=X,
    y_train=y,
    x_query=x_query,   # user-controlled input
    yhat=yhat,
    title="Synthetic example: 15 variables → target"
)

fig.show()


# -------------------------
# Synthetic data: 30 features
# -------------------------
np.random.seed(7)

n = 300        # number of samples
d = 30         # number of features

# Generate input matrix
X = np.random.normal(0, 1.0, size=(n, d))

# True underlying parameters (only first few nonzero, rest small noise)
true_w = np.zeros(d)
true_w[:5] = [2.5, -1.7, 0.9, 0.0, -1.2]   # meaningful coefficients
true_w[5:] = np.random.uniform(-0.2, 0.2, size=d-5)

true_b = 0.7

# Generate target with noise
y = X @ true_w + true_b + np.random.normal(0, 0.8, size=n)

print("d =", X.shape[1])

# ------------------------------------------------------------
# User-chosen query point (like sklearn.predict)
# Here we simply take one realistic sample from X
# ------------------------------------------------------------
i = 0
x_query = X[i:i+1]   # shape (1, 30)

# ============================================================
# Model: SGDRegressor + StandardScaler (optional but realistic)
# ============================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.02,
        shuffle=False,
        max_iter=40000,
        tol=1e-8,
        random_state=7,
    ))
])

model.fit(X, y)

# Standard sklearn prediction
yhat = float(model.predict(x_query)[0])

# ============================================================
# Visualization with your explainer
# ============================================================
fig = explain_lr_prediction(
    trained_estimator=model,
    X_train=X,
    y_train=y,
    x_query=x_query,   # user-controlled input
    yhat=yhat
)

fig.show()
