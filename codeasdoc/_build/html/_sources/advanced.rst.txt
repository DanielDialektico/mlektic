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
