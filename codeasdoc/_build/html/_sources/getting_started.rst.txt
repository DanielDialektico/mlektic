========================
Inicio Rápido
========================

Instalación
===========

El proyecto utiliza ``uv`` como gestor de dependencias (compatible con PEP 621).

.. code-block:: bash

   git clone https://github.com/DanielDialektico/mlektic.git
   cd mlektic
   uv sync

Dependencias principales:

- **numpy** — Álgebra lineal y manipulación de arrays.
- **scikit-learn** — Modelos de Machine Learning.
- **plotly** — Motor de visualización interactiva.

La integración con PyTorch es opcional:

.. code-block:: bash

   pip install "mlektic[torch]"

Primer Ejemplo: Regresión Lineal
================================

.. code-block:: python

   import numpy as np
   import plotly.io as pio
   from sklearn.linear_model import SGDRegressor
   from mlektic import visualize_lr

   pio.renderers.default = "notebook"

   # Datos de juguete
   X = np.sort(np.random.rand(100, 1)) * 10
   y = 2.5 * X.ravel() + 1.0 + np.random.randn(100) * 2

   # Modelo
   model = SGDRegressor(
       loss="squared_error",
       max_iter=50,
       learning_rate="constant",
       eta0=0.005,
       random_state=42,
   )
   model.fit(X, y)

   # Animación
   fig = visualize_lr(
       model, X, y,
       steps=60,
       animation_mode="hybrid",
       fps=30,
       interpolation_frames=3,
       show_loss=True,
       title="Mi Primera Animación Mlektic",
   )
   fig.show()

Si la animación en el editor es lenta, puedes exportarla a HTML:

.. code-block:: python

   fig.write_html("animacion.html", auto_play=False)


Primer Ejemplo: Regresión Logística
====================================

.. code-block:: python

   import numpy as np
   import plotly.io as pio
   from sklearn.linear_model import SGDClassifier
   from mlektic import visualize_logistic

   pio.renderers.default = "notebook"

   # Datos binarios
   np.random.seed(42)
   X = np.random.randn(200, 1)
   y = (X.ravel() > 0).astype(int)

   model = SGDClassifier(
       loss="log_loss",
       learning_rate="constant",
       eta0=0.05,
       max_iter=500,
       random_state=42,
   )
   model.fit(X, y)

   fig = visualize_logistic(
       model, X, y,
       steps=60,
       show_loss=True,
       frame_duration=80,
       transition_duration=70,
       title="Regresión Logística Binaria",
   )
   fig.show()


Primer Ejemplo: Red Neuronal PyTorch
====================================

El recorder debe guardar cada frame después de ``optimizer.step()`` y antes del
siguiente ``zero_grad()``. Así conserva el peso actualizado junto con el gradiente
que lo produjo.

.. code-block:: python

   import torch
   from mlektic import TorchTrainingRecorder, visualize_nn_graph, visualize_nn_training

   X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
   y = torch.tensor([[0.], [1.], [1.], [0.]])
   model = torch.nn.Sequential(
       torch.nn.Linear(2, 4),
       torch.nn.Tanh(),
       torch.nn.Linear(4, 1),
       torch.nn.Sigmoid(),
   )
   optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
   loss_fn = torch.nn.BCELoss()
   recorder = TorchTrainingRecorder(model, optimizer=optimizer, loss_fn=loss_fn)

   for step in range(80):
       optimizer.zero_grad()
       prediction = model(X)
       loss = loss_fn(prediction, y)
       loss.backward()
       optimizer.step()

       with torch.no_grad():
           prediction = model(X)
           recorded_loss = loss_fn(prediction, y)
       recorder.record(
           step,
           loss=recorded_loss,
           predictions=prediction,
           targets=y,
           task="classification",
       )

   history = recorder.to_history()
   recorder.close()

   visualize_nn_graph(model, X[0], history).show()
   visualize_nn_training(history).show()

``record`` infiere ``accuracy``, ``precision`` macro y ``recall`` macro para
clasificación. Para regresión, ``task="regression"`` produce MSE, MAE y R2.


Uso con Pipelines
=================

Mlektic soporta nativamente ``sklearn.pipeline.Pipeline`` con pasos de
preprocesamiento como ``StandardScaler``:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   model = Pipeline([
       ("scaler", StandardScaler()),
       ("sgd", SGDRegressor(
           loss="squared_error",
           learning_rate="constant",
           eta0=0.001,
           max_iter=2000,
       )),
   ])
   model.fit(X, y)

   # Visualizar en espacio original (des-transforma θ automáticamente)
   fig = visualize_lr(model, X, y, display_space="original")

   # Visualizar en espacio escalado (θ tal cual los aprende el modelo)
   fig_scaled = visualize_lr(model, X, y, display_space="scaled")
