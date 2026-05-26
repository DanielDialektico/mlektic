import sys
from pathlib import Path
import urllib.request
import io

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic.api.linear import explain_lr_prediction

# -------------------------
# Real data: Advertising dataset (TV, Radio -> Sales)
# Source CSV (downloadable):
# https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv
# -------------------------
url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
response = urllib.request.urlopen(url)
data = np.genfromtxt(response, delimiter=',', skip_header=1)

# Columns: Unnamed: 0, TV, Radio, Newspaper, Sales
# Use 2 features: TV + Radio (indices 1, 2)
X = data[:, 1:3]      # TV, Radio (n, 2)
y = data[:, 4]        # Sales (n,)

# ------------------------------------------------------------
# 1) Modelo DIRECTO (sin scaler) - 2 variables
# ------------------------------------------------------------
model_raw = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.02,
    shuffle=False,
    max_iter=40000,
    tol=1e-8,
    random_state=7,
)
model_raw.fit(X, y)

# Punto que tú decides, en unidades ORIGINALES (como sklearn.predict)
# Ejemplo: TV=150, Radio=25
x_query = np.array([[150.0, 25.0]], dtype=float)

# Ideal: usa predict() y pásalo tal cual al explainer
yhat_raw = float(model_raw.predict(x_query)[0])

fig = explain_lr_prediction(
    model_raw, X, y,
    x_query=x_query,
    yhat=yhat_raw,
    title="Advertising (SGDRegressor): (TV, Radio) → Sales (raw)",
    display_space="original",
)
fig.show()

# ------------------------------------------------------------
# 2) Pipeline con StandardScaler + SGDRegressor - 2 variables
# ------------------------------------------------------------
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
    )),
])
model_scaled.fit(X, y)

# Mismo punto, SIEMPRE en unidades ORIGINALES al llamar predict()
yhat_scaled = float(model_scaled.predict(x_query)[0])  # Sales (no escalado)

# a) Mostrar explicación en espacio ORIGINAL (x original + theta convertida)
fig = explain_lr_prediction(
    model_scaled, X, y,
    x_query=x_query,
    yhat=yhat_scaled,
    title="Advertising (Pipeline: StandardScaler + SGDRegressor): (TV, Radio) → Sales (display original)",
    display_space="original",
)
fig.show()

# b) Mostrar explicación en espacio ESCALADO (x escalada + theta escalada)
fig = explain_lr_prediction(
    model_scaled, X, y,
    x_query=x_query,
    yhat=yhat_scaled,
    title="Advertising (Pipeline: StandardScaler + SGDRegressor): (TV, Radio) → Sales (display scaled)",
    display_space="scaled",
)
fig.show()
