==============================
Sistema de Visualización
==============================

Mlektic genera figuras Plotly animadas con un tema visual cohesivo en **dark mode**.

Redes Neuronales PyTorch
========================

La familia neural es opcional y se instala con ``pip install "mlektic[torch]"``.
Su API esta orientada a notebooks y Colab. Cada vista conserva el hilo matematico
entre arquitectura, entrenamiento y prediccion.

- ``visualize_nn_architecture`` representa cada tipo de capa con una forma semantica
  y muestra formulas LaTeX, dimensiones de entrada/salida, parametros e
  hiperparametros. En arquitecturas extensas intercala puntos suspensivos.
- ``visualize_nn_graph`` crea un frame estable por paso. Las conexiones forman un
  mapa de calor global entre el peso minimo y maximo; las lineas punteadas color
  tinto superpuestas representan la magnitud del gradiente de backpropagation. El
  color de cada nodo representa su salida numerica exacta para la entrada elegida,
  usando por defecto una sola escala global con los minimos y maximos reales de
  todos los nodos y frames, incluidos valores negativos. El hover conserva la
  salida exacta y evita mostrar sintaxis LaTeX cruda. Las aristas codifican
  :math:`\theta_{ji}` y los nodos
  :math:`a_j=\phi(\sum_i\theta_{ji}a_i+\theta_{0,j})`, por lo que
  usan escalas distintas. El ultimo frame usa los tensores finales del modelo.
  ``node_color_mode="relative"`` ofrece contraste normalizado por capa como modo
  opcional. ``edge_color_mode="signal"`` colorea cada arista mediante la
  contribucion forward :math:`\theta_{ji}a_i`; el modo predeterminado ``"weight"``
  conserva la visualizacion de los parametros. El hover diferencia ``0 (exact)``
  de ``0 (ReLU inactive)`` y usa notacion cientifica para valores no nulos que
  serian redondeados a ``0.000``. Los valores compactos :math:`\Theta_t` y el paso
  temporal son trazas animadas, por lo que avanzan sin redibujar ni hacer
  parpadear la red.
- ``TorchTrainingRecorder`` registra loss, metricas proporcionadas por el usuario,
  normas L2, gradientes, vectores de activacion compactos y snapshots de tensores
  pequenos. Tambien puede inferir tres metricas al recibir ``predictions`` y
  ``targets``: accuracy, precision macro y recall macro para clasificacion; MSE,
  MAE y R2 para regresion. ``record`` se llama despues de ``optimizer.step()`` y
  antes del siguiente ``zero_grad()`` para conservar tanto el peso actualizado
  como el gradiente que lo origino.
- ``visualize_nn_training`` distribuye loss y tres metricas de rendimiento en una
  cuadricula compacta de 2 por 2. Si el historial no contiene metricas, mantiene
  los cuatro paneles e indica como registrarlas. ``visualize_nn_weights`` muestra matrices LaTeX con
  definiciones, dimensiones y truncado explicito. La vista ``activations`` muestra
  formulas, vectores y estadisticas compactas por capa.
- ``explain_nn_prediction`` anima la composicion de funciones y sustituye valores
  numericos en ``z = Theta a + theta_0``. En redes profundas conserva primeras y ultimas capas
  y marca el tramo omitido con puntos suspensivos.
- ``display_nn_math_report`` inserta en Jupyter un reporte con la taxonomia completa.
  ``export_nn_math_report`` genera el mismo documento HTML independiente, con una
  seccion por capa, configuracion de entrenamiento y evolucion de parametros.

Los limites ``max_frames``, ``max_layers``, ``max_neurons``, ``max_rows`` y
``max_cols`` mantienen responsivas las figuras. Los tensores mayores que
``max_tensor_elements`` no se duplican en el historial, aunque sus normas y metricas
si se conservan.

Ejemplo minimo
--------------

.. code-block:: python

   recorder = TorchTrainingRecorder(
       model,
       optimizer=optimizer,
       loss_fn=loss_fn,
       record_every=2,
   )

   # Dentro del entrenamiento, despues de optimizer.step y antes de zero_grad:
   recorder.record(
       step,
       loss=loss,
       predictions=prediction,
       targets=y,
       task="classification",
   )

   history = recorder.to_history()
   visualize_nn_architecture(model, X[:1], history=history).show()
   visualize_nn_graph(model, X[0], history, max_frames=16).show()
   visualize_nn_training(history, max_frames=24).show()
   visualize_nn_weights(history, max_rows=4, max_cols=5).show()
   explain_nn_prediction(model, X[0], history=history).show()
   report_path = export_nn_math_report(
       model,
       X[:1],
       history=history,
       path="network-report.html",
   )

Modos del grafo matematico
--------------------------

La configuracion predeterminada conserva unidades y valores reales en dos escalas
globales independientes:

.. code-block:: python

   visualize_nn_graph(
       model,
       X[0],
       history,
       node_color_mode="value",  # a_j real; default
       edge_color_mode="weight", # w_ji real; default
   ).show()

La escala de nodos no debe compararse numericamente con la de aristas:

.. math::

   \text{peso}=\theta_{ji},\qquad
   \text{senal}=\theta_{ji}a_i,\qquad
   a_j=\phi\!\left(\sum_i\theta_{ji}a_i+\theta_{0,j}\right).

Para resaltar cambios pequenos por capa puede usarse una escala relativa. Para
visualizar lo que aporta cada conexion durante el forward pass puede colorearse
por senal:

.. code-block:: python

   visualize_nn_graph(
       model,
       X[0],
       history,
       node_color_mode="relative",
       edge_color_mode="signal",
   ).show()

``relative`` mejora el contraste, pero deja de expresar una escala absoluta comun.
En ambos modos el hover conserva la salida, el peso, la senal transmitida, el
gradiente y la actualizacion exactos.

Reportes HTML en notebook
-------------------------

Para redes complejas, el reporte completo puede mostrarse directamente en Jupyter
o Colab:

.. code-block:: python

   from IPython.display import display
   from mlektic import display_nn_math_report

   display(
       display_nn_math_report(
           model,
           X[:1],
           history=history,
           title="Complex network mathematics",
       )
   )

Si el archivo ya fue exportado, puede insertarse en Jupyter con
``HTML(filename="network-report.html")`` o ``IFrame``. En Colab se recomienda
``display_nn_math_report`` porque no depende de que el navegador pueda resolver
una ruta local.

Tema Visual Global
===================

Las figuras Scikit-Learn comparten ``visualization/theme.py`` y las vistas neurales
usan el lenguaje equivalente definido en ``visualization/neural/_style.py``:

- **Template base**: ``plotly_dark`` con tipografia clara de alto contraste.
- **Altura**: 720px por defecto.
- **Controles**: Botones Play/Pause y slider temporal integrados.
- **Leyenda**: Fondo semi-transparente con texto negro sobre blanco.

Renderizado por Dimensión
==========================

d = 1: Regresión Simple (2D)
------------------------------

- **Lineal**: Scatter plot con recta animada + curva MSE opcional.
- **Logística Binaria**: Scatter coloreado + curva sigmoide animada.
- **Logística Multiclase**: Curvas de probabilidad por clase.

d = 2: Visualización 3D
-------------------------

- **Lineal**: Scatter 3D + plano predictivo animado.
- **Logística Binaria**: Scatter 3D + superficie de probabilidad.
- **Logística Multiclase**: Scatter 3D + ``K`` superficies de probabilidad y un
  panel LaTeX con :math:`\Theta\in\mathbb{R}^{2\times K}`,
  :math:`\boldsymbol{\theta}_0\in\mathbb{R}^{K}` y el enlace real del estimador.
  ``multiclass_link="auto"`` distingue Softmax multinomial de OvR mediante
  sigmoides normalizadas; ``"softmax"`` y ``"ovr"`` permiten forzarlo.

d > 2: Matriz de Parámetros LaTeX
-----------------------------------

- Tabla/fórmula LaTeX interactiva con ``θ`` actualizado por frame.
- Multiclase: :math:`\Theta\in\mathbb{R}^{d\times K}` y
  :math:`\boldsymbol{\theta}_0\in\mathbb{R}^{K}`, aunque la matriz se trunque o
  redistribuya visualmente.

Estrategias de Captura
=======================

Iterativa (``mode="iterative"``)
---------------------------------

Para modelos con ``partial_fit`` / ``warm_start``:

1. Clona el estimador con ``warm_start=True``, ``max_iter=1``.
2. Ejecuta una iteración por frame.
3. Captura predicciones, pesos, pérdida, y, en configuraciones multiclase 2D, extrae un historial matricial de superficies probabilísticas (``p_surfaces_hist``).

Interpolación (``mode="final_interp"``)
-----------------------------------------

Para modelos sin entrenamiento iterativo (e.g. ``LinearRegression``):

1. Define línea base (media, ceros, prior).
2. Interpola linealmente hacia predicciones finales.

Modo Automático (``mode="auto"``)
-----------------------------------

Detecta ``partial_fit`` / ``warm_start`` → iterativa; si no → interpolación.

Suavizado EMA
==============

Cuando ``smooth="ema"``:

.. math::

   \widetilde{\mathcal{L}}_t = \beta \widetilde{\mathcal{L}}_{t-1}
   + (1 - \beta)\mathcal{L}_t

``smooth_beta`` controla la agresividad (0.95 = muy suave, 0.5 = más detalle).
El suavizado solo afecta a la pérdida: las predicciones, probabilidades y
parámetros conservan los checkpoints exactos para no romper su correspondencia
matemática.

Transiciones y muestreo temporal
================================

``steps`` controla los estados capturados; ``max_frames`` y ``frame_step``
reducen el historial sin perder el primer ni el ultimo estado. Para trazas 2D,
``transition_duration`` interpola visualmente entre frames y puede fijarse en
``0`` para desactivarla. Plotly redibuja las superficies 3D porque no admite la
misma interpolacion estable; en ese caso la suavidad depende principalmente de
``steps``, ``max_frames`` y ``frame_duration``.

Animación híbrida 1D
====================

En regresión lineal de una variable, ``animation_mode="auto"`` selecciona el
modo híbrido. ``steps`` y ``max_frames`` siguen describiendo checkpoints reales;
``interpolation_frames`` agrega subframes visuales entre ellos y ``fps`` controla
su cadencia. La recta, la pérdida, las métricas y los coeficientes numéricos usan
el mismo parámetro interpolado. La definición simbólica permanece en LaTeX y no
obliga a redibujar el layout, por lo que funciona con ``redraw=False`` en Jupyter,
Colab y HTML exportado.

Si ``fps`` es ``None``, la duración visual se calcula como
``frame_duration / interpolation_frames``. Por ejemplo, ``30 / 3 = 10 ms``
solicita 100 FPS, una cadencia que suele superar la capacidad de Jupyter y causa
cuadros descartados. Para notebooks se recomienda ``fps=30`` a ``45``, o
``frame_duration=60`` a ``80`` con tres subframes.

El texto numérico del modo híbrido usa una fuente matemática dentro de una traza:
MathJax no puede interpolar glifos LaTeX entre frames. ``animation_mode="native"``
conserva la sustitución dinámica en LaTeX, pero cada cambio obliga a redibujar el
layout. La regresión logística 1D utiliza actualmente este mecanismo nativo; una
sigmoide acotada puede percibirse fluida con suficientes frames, aunque sus
fórmulas se sustituyen discretamente y no se interpolan.

Las métricas se apilan en una columna lateral independiente para que sus valores
no compitan por espacio con la curva de pérdida. Play y Pause permanecen blancos
con texto negro en reposo, hover y reproducción, incluso si Plotly reconstruye
los controles durante una animación 3D.

Espacio de Visualización
=========================

Con ``Pipeline`` + ``StandardScaler``:

- ``display_space="scaled"`` → ``θ`` tal como los aprende el modelo.
- ``display_space="original"`` → Transforma inversamente los pesos a las unidades originales.

Explicación Visual de Predicciones
===================================

Mediante las funciones ``explain_lr_prediction`` y ``explain_logistic_prediction``, Mlektic permite diseccionar matemáticamente cómo un modelo generó una predicción puntual ``yhat`` a partir de un ``x_query``:

- **1D**: Punto verde sobre la recta de regresión.
- **2D**: Punto verde anclado espacialmente en un plano 3D.
- **ND**: Presentación dinámica de fórmulas en LaTeX que despliega el producto
  punto mediante vectores columna. ``\vdots`` omite componentes intermedias sin
  cambiar su orden ni introducir semántica matricial.

Las vistas de predicción usan notación científica LaTeX para magnitudes grandes,
por ejemplo :math:`1.6609\times 10^7`, y reservan tamaños tipográficos menores en
los paneles de sustitución y resultado de dos variables.

Métricas Dinámicas
===================

Las visualizaciones animadas pueden renderizar arreglos personalizables de
métricas en tiempo real.

- Regresión lineal: ``metrics=["loss", "mse", "r2", "mae"]``.
- Regresión logística: ``metrics=["loss", "accuracy", "f1"]``.
- Métricas personalizadas: ``metrics={"Nombre": callable}``, donde ``callable``
  recibe ``(y_true, y_pred)`` y devuelve un escalar.

Mlektic computa y muestra estas métricas como subtítulos o recuadros separados.
La animación híbrida 1D coloca sus recuadros en una columna vertical dedicada.
