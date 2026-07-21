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

Las trazas 2D interpolan entre estados mediante ``transition_duration``. Un valor
``None`` elige automaticamente una duracion menor que ``frame_duration`` para que
la interpolacion termine antes del siguiente frame; los valores iguales o mayores
tambien se limitan de forma segura. ``0`` desactiva la transicion. Las superficies
3D requieren redibujado y obtienen su continuidad aumentando ``steps`` o el limite
``max_frames``.

Las lineas animadas conservan todos sus puntos durante la interpolacion. Esto evita
que la simplificacion geometrica de Plotly cambie la ruta SVG entre frames y produzca
segmentos parciales o parpadeos en renderizadores de Jupyter.

Cuando un frame modifica ecuaciones LaTeX o tarjetas de metricas, la figura activa
automaticamente el redibujado de layout requerido por Jupyter. Las animaciones que
solo cambian trazas conservan la actualizacion ligera sin redibujado completo. En
frames mixtos, el orden ``traces first`` interpola primero la recta o curva y aplica
despues el nuevo estado matematico, evitando que el layout anule la fluidez visual.

MathJax sustituye una expresion LaTeX completa; no interpola los digitos entre dos
expresiones. Por eso una formula LaTeX que parece fluida es una secuencia densa de
estados discretos. En curvas acotadas, como la sigmoide logistica, el redibujado
puede ser menos perceptible que en una recta que recorre un rango amplio.

Para regresion lineal 1D, ``animation_mode="auto"`` usa una estrategia hibrida:
la formula simbolica queda fija en LaTeX y los coeficientes numericos, la recta,
la perdida y las metricas avanzan como trazas sincronizadas. ``fps`` fija la
cadencia y ``interpolation_frames`` controla cuantos intervalos visuales existen
entre checkpoints. El slider sigue mostrando solo pasos semanticos reales.
Sin ``fps``, cada subframe dura ``frame_duration / interpolation_frames``. Evita
valores inferiores a unos 16 ms en el navegador; para Jupyter y Colab, ``fps=30``
a ``45`` suele ofrecer una cadencia mas estable que solicitar 100 FPS.

En clasificacion multiclase, ``multiclass_link="auto"`` compara los scores con
``predict_proba`` para distinguir Softmax de sigmoides OvR normalizadas. El
override ``"softmax"`` o ``"ovr"`` resulta util para estimadores personalizados.


Control de Redes PyTorch
========================

``TorchTrainingRecorder`` limita por defecto los tensores completos a 4096
elementos. Los tensores mayores conservan sus normas sin duplicarse en el historial.
Para una red pequena cuyo grafo matematico deba mostrar todos los pesos, aumenta el
limite de forma consciente:

.. code-block:: python

   recorder = TorchTrainingRecorder(
       model,
       max_tensor_elements=10_000,
       max_activation_elements=512,
       record_every=2,
   )

``max_frames`` selecciona pasos distribuidos uniformemente sin inventar estados
intermedios. El ultimo frame siempre usa los tensores actuales del modelo:

.. code-block:: python

   fig = visualize_nn_graph(
       model,
       X[0],
       history,
       max_frames=24,
       frame_duration=180,
       node_color_mode="value",
       edge_color_mode="weight",
   )

El modo ``value`` usa un minimo y maximo global reales para todos los nodos y
frames mostrados. Un nodo gris representa una salida cercana a cero dentro de esa
escala; no implica que el nodo sea constante. El hover distingue ceros exactos,
ReLU inactivas y valores pequenos expresados en notacion cientifica.

Para inspeccionar el flujo en vez del parametro aislado:

.. code-block:: python

   signal_fig = visualize_nn_graph(
       model,
       X[0],
       history,
       edge_color_mode="signal",  # w_ji * a_i
   )

La contribucion :math:`\theta_{ji}a_i` tampoco coincide necesariamente con la salida del
nodo receptor, que agrega todas las entradas, suma el bias y aplica la activacion.

Exportar a HTML
================

Para compartir animaciones sin depender de Jupyter:

.. code-block:: python

   fig.write_html("mi_animacion.html", auto_play=False)

El archivo HTML es autocontenido y puede abrirse en cualquier navegador.

El reporte matematico completo de una red usa una API separada:

.. code-block:: python

   from IPython.display import display
   from mlektic import display_nn_math_report, export_nn_math_report

   display(display_nn_math_report(model, X[:1], history=history))
   path = export_nn_math_report(
       model,
       X[:1],
       history=history,
       path="complex-network-mathematics.html",
   )

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

La API tabular de alto nivel trabaja contra adapters. Para soportar otro framework,
el adapter debe traducir su modelo al contrato interno: prediccion, probabilidades,
extraccion de parametros cuando aplique, captura incremental o replay, y datos de
escalado. PyTorch ya cuenta con una integracion especializada mediante
``TorchTrainingRecorder`` y builders neurales; otros frameworks pueden implementar
un recorder equivalente o adaptarse al contrato tabular cuando corresponda.
