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
- ``visualize_nn_graph`` crea dos frames por paso: ``F`` para feed forward y ``B``
  para backpropagation. El color codifica signo y magnitud; el hover muestra pesos,
  activaciones, bias, gradientes, actualizaciones y dimensiones exactas.
- ``TorchTrainingRecorder`` registra loss, metricas proporcionadas por el usuario,
  normas L2, gradientes, vectores de activacion compactos y snapshots de tensores
  pequenos. ``record`` se llama despues de ``loss.backward()`` y antes de
  ``optimizer.step()`` para conservar el gradiente que origina la actualizacion.
- ``visualize_nn_training`` separa la funcion de perdida de las metricas de
  rendimiento. ``visualize_nn_weights`` reemplaza el heatmap por matrices LaTeX con
  definiciones, dimensiones y truncado explicito. La vista ``activations`` muestra
  formulas, vectores y estadisticas compactas por capa.
- ``explain_nn_prediction`` anima la composicion de funciones y sustituye valores
  numericos en ``z = Wa + b``. En redes profundas conserva primeras y ultimas capas
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

   # Dentro del entrenamiento, despues de backward y antes de optimizer.step:
   recorder.record(step, loss=loss, metrics={"accuracy": accuracy})

   history = recorder.to_history()
   visualize_nn_architecture(model, X[:1], history=history).show()
   visualize_nn_graph(model, X[0], history, max_frames=16).show()
   visualize_nn_training(history, max_frames=24).show()
   visualize_nn_weights(history, max_rows=4, max_cols=5).show()
   explain_nn_prediction(model, X[0], history=history).show()
   export_nn_math_report(model, X[:1], history=history, path="network-report.html")

Tema Visual Global
===================

Todas las figuras comparten un tema definido en ``visualization/theme.py``:

- **Template base**: ``plotly_dark`` con fuente Helvetica blanca.
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
- **Logística Multiclase**: Scatter 3D + ``K`` superficies de probabilidad translúcidas superpuestas, acompañadas de un panel LaTeX que muestra dinámicamente la matriz de pesos :math:`\Theta \in \mathbb{R}^{3 \times K}` y las ecuaciones softmax evaluadas.

d > 2: Matriz de Parámetros LaTeX
-----------------------------------

- Tabla/fórmula LaTeX interactiva con ``θ`` actualizado por frame.
- Multiclase: Matriz ``W`` (d × K) y vector ``b`` (K).

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

   y_t = \beta \cdot y_{t-1} + (1 - \beta) \cdot x_t

``smooth_beta`` controla la agresividad (0.95 = muy suave, 0.5 = más detalle).

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
- **ND**: Presentación dinámica de fórmulas en LaTeX que despliega el producto punto completo, truncando automáticamente parámetros desbordados y manejando el espacio dimensional especificado.

Métricas Dinámicas
===================

Las visualizaciones animadas pueden renderizar arreglos personalizables de
métricas en tiempo real.

- Regresión lineal: ``metrics=["loss", "mse", "r2", "mae"]``.
- Regresión logística: ``metrics=["loss", "accuracy", "f1"]``.
- Métricas personalizadas: ``metrics={"Nombre": callable}``, donde ``callable``
  recibe ``(y_true, y_pred)`` y devuelve un escalar.

Mlektic computa y muestra estas métricas como subtítulos o recuadros separados,
según el builder de visualización usado.
