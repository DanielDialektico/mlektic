==============================
Sistema de Visualización
==============================

Mlektic genera figuras Plotly animadas con un tema visual cohesivo en **dark mode**.

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
3. Captura predicciones, pesos, pérdida.

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

Mediante la función ``explain_lr_prediction``, Mlektic permite diseccionar matemáticamente cómo un modelo generó una predicción puntual ``yhat`` a partir de un ``x_query``:

- **1D**: Punto verde sobre la recta de regresión.
- **2D**: Punto verde anclado espacialmente en un plano 3D.
- **ND**: Presentación dinámica de fórmulas en LaTeX que despliega el producto punto completo, truncando automáticamente parámetros desbordados y manejando el espacio dimensional especificado.

Métricas Dinámicas
===================

Las visualizaciones animadas pueden renderizar arreglos personalizables de métricas en tiempo real pasando una lista a ``metrics=["loss", "mse", "r2", ...]``. Mlektic computará y mostrará estas métricas como subtítulos en recuadros separados.
