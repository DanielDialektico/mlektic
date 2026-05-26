import sys
from pathlib import Path

# Añadimos 'src' al path para poder importar mlektic localmente sin instalar
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlektic.api.linear import explain_lr_prediction


# -------------------------
# Real data: Advertising dataset (TV -> Sales)
# Source CSV (downloadable):
# https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv
# -------------------------
import urllib.request
import io

url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
response = urllib.request.urlopen(url)
data = np.genfromtxt(response, delimiter=',', skip_header=1)

# Columns: Unnamed: 0, TV, Radio, Newspaper, Sales
X = data[:, 1:2]      # TV (n, 1)
y = data[:, 4]        # Sales (n,)

# -------------------------
# 1) Modelo DIRECTO (sin scaler)
# -------------------------
model_raw = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=0.00001, # Ajustado eta0 para evitar que explote con TV (valores de 0 a 300)
    shuffle=False,
    max_iter=20000,
    tol=1e-8,
    random_state=7,
)
model_raw.fit(X, y)

# Punto que tú decides (como sklearn.predict)
x_query = np.array([[150.0]], dtype=float)   # TV=150 (miles de dólares)
yhat_raw = model_raw.predict(x_query)[0]

fig = explain_lr_prediction(
    model_raw, X, y,
    x_query=x_query,
    yhat=yhat_raw,  # recomendado: tomamos lo que dio predict()
    title="Advertising (SGDRegressor): TV → Sales (raw)",
    display_space="original",
)
fig.show()


# -------------------------
# 2) Pipeline con StandardScaler + SGDRegressor
# -------------------------
model_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("sgd", SGDRegressor(
        loss="squared_error",
        penalty=None,
        learning_rate="constant",
        eta0=0.02,
        shuffle=False,
        max_iter=20000,
        tol=1e-8,
        random_state=7,
    ))
])
model_scaled.fit(X, y)

# Mismo punto en unidades originales (TV=150)
x_query = np.array([[150.0]], dtype=float)
yhat_scaled = model_scaled.predict(x_query)[0]  # salida SIEMPRE en unidades de Sales (no escaladas)

# Verlo en espacio ORIGINAL (x original + theta convertida a original si hay scaler)
fig1 = explain_lr_prediction(
    model_scaled, X, y,
    x_query=x_query,
    yhat=yhat_scaled,
    title="Advertising (Pipeline): TV → Sales (display original)",
    display_space="original",
)
fig1.show()

# Verlo en espacio ESCALADO (x escalada + theta en espacio escalado)
fig2 = explain_lr_prediction(
    model_scaled, X, y,
    x_query=x_query,
    yhat=yhat_scaled,
    title="Advertising (Pipeline): TV → Sales (display scaled)",
    display_space="scaled",
)
fig2.show()
