========================
Uso Avanzado
========================

Uso Granular de la API
=======================

Puedes usar las funciones de bajo nivel por separado para mayor control:

.. code-block:: python

   from mlektic import fit_history, build_lr_figure

   # Capturar historial
   history = fit_history(
       model, X, y,
       steps=80,
       mode="iterative",
       smooth="ema",
       smooth_beta=0.9,
       baseline="zeros",
       display_space="original",
       metrics=["loss", "mse", "r2", "mae"],
       max_frames=60,
   )

   # Construir figura con opciones
   fig = build_lr_figure(
       X, y,
       history=history,
       show_loss=True,
       title="Mi Gráfico Personalizado",
       dec=6,
       frame_duration=50,
   )


Métricas Personalizadas
=======================

Las visualizaciones calculan métricas por frame para alimentar subtítulos y
paneles de la animación.

En regresión lineal puedes solicitar métricas integradas:

.. code-block:: python

   fig = visualize_lr(
       model, X, y,
       metrics=["loss", "mse", "r2", "mae"],
   )

En regresión logística están disponibles:

.. code-block:: python

   fig = visualize_logistic(
       model, X, y,
       metrics=["loss", "accuracy", "f1"],
   )

También puedes pasar funciones personalizadas como diccionario. Cada función
recibe ``(y_true, y_pred)`` y debe devolver un escalar:

.. code-block:: python

   history = fit_history(
       model, X, y,
       metrics={
           "Error mediano": lambda y_true, y_pred: np.median(np.abs(y_true - y_pred)),
       },
   )


Control de Frames
=================

Para entrenamientos largos, ``steps`` puede ser mayor que la cantidad de frames
que quieres renderizar. ``max_frames`` reduce el historial de forma uniforme
antes de construir la figura:

.. code-block:: python

   fig = visualize_lr(model, X, y, steps=500, max_frames=80)

Si necesitas muestrear cada N pasos en vez de fijar un máximo, desactiva
``max_frames`` y usa ``frame_step``:

.. code-block:: python

   fig = visualize_logistic(model, X, y, steps=500, max_frames=None, frame_step=20)

Exportar a HTML
================

Para compartir animaciones sin depender de Jupyter:

.. code-block:: python

   fig.write_html("mi_animacion.html", auto_play=False)

El archivo HTML es autocontenido y puede abrirse en cualquier navegador.

Configuración del Renderer
============================

Plotly requiere configurar el renderer según tu entorno:

.. code-block:: python

   import plotly.io as pio

   # Jupyter Notebook / VS Code
   pio.renderers.default = "notebook"

   # Google Colab
   pio.renderers.default = "colab"

   # Abrir en navegador
   pio.renderers.default = "browser"

.. note::
   Mlektic establece ``pio.renderers.default = "colab"`` al importar.
   Si usas otro entorno, sobreescríbelo antes de llamar a ``fig.show()``.


Dimensionalidad Alta
=====================

Para datasets con muchas features (d > 2), Mlektic genera una representación
LaTeX del vector de parámetros. Ejemplo con 150 dimensiones:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import SGDRegressor
   from mlektic import visualize_lr

   model = Pipeline([
       ("scaler", StandardScaler()),
       ("sgd", SGDRegressor(eta0=0.015, max_iter=1000)),
   ])
   model.fit(X_150d, y)

   fig = visualize_lr(
       model, X_150d, y,
       steps=60,
       show_loss=True,
       display_space="original",
   )


Directorio ``local_test/``
===========================

Scripts preconfigurados para probar todas las capacidades:

- ``test_1_var.py`` — Regresión lineal 1D con datos reales y sintéticos.
- ``test_2_vars.py`` — Regresión lineal 2D con Pipeline.
- ``test_multivar_pipeline.py`` — 8, 100 y 150 dimensiones.
- ``test_refactor_linear.py`` — Tests con ``LinearRegression`` (interpolación).
- ``test_log_var.py`` — Regresión logística binaria con Breast Cancer.

Ejecución:

.. code-block:: bash

   cd local_test
   python test_1_var.py


Compatibilidad con Futuros Adapters
===================================

La API de alto nivel trabaja contra adapters. Para soportar otro framework, el
adapter debe traducir su modelo al contrato interno: predicción, probabilidades,
extracción de parámetros cuando aplique, captura incremental o replay, y datos
de escalado. Esta separación deja la librería preparada para modelos no
Scikit-Learn y para una futura capa de visualización de redes neuronales sin
acoplar cada figura a un framework específico.
