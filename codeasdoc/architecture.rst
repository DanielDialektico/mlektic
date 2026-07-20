==========================
Arquitectura del Proyecto
==========================

La integracion PyTorch vive en dos paquetes independientes del flujo Scikit-Learn.
``neural/`` contiene introspeccion, la taxonomia matematica reutilizable,
``TorchTrainingRecorder`` y el generador de reportes HTML. ``visualization/neural/``
contiene builders separados para arquitectura, grafo temporal, entrenamiento,
parametros, activaciones y sustituciones del forward pass. PyTorch se importa solo al
invocar estas funciones, por lo que sigue siendo una dependencia opcional.

Los builders consumen diccionarios de descripcion e historial, no estado visual
global. Los limites ``max_frames``, ``max_layers``, ``max_neurons`` y
``max_tensor_elements`` permiten resumir redes grandes sin cambiar el contrato. Esta
separacion permite sumar CNNs, grafos FX o adapters de otros frameworks reutilizando
las definiciones y las politicas de truncado.

Mlektic sigue una **arquitectura hexagonal** (Ports & Adapters) simplificada, organizada
en capas con responsabilidades claras.

Diagrama General
================

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    API Pública                          │
   │        visualize_lr()    visualize_logistic()           │
   └──────────────┬─────────────────────┬────────────────────┘
                  │                     │
   ┌──────────────▼─────────────────────▼────────────────────┐
   │                    Services                             │
   │      fit_history()     fit_history_logistic()           │
   └──────────────┬─────────────────────┬────────────────────┘
                  │                     │
   ┌──────────────▼─────────────────────▼────────────────────┐
   │                  History Engine                         │
   │    HistoryEngine → Strategy Pattern                     │
   │    ├── IterativeCapture  (partial_fit / warm_start)     │
   │    └── InterpolationCapture  (modelos no iterativos)    │
   └──────────────┬─────────────────────┬────────────────────┘
                  │                     │
   ┌──────────────▼──────┐  ┌──────────▼────────────────────┐
   │      Adapters       │  │      Visualization            │
   │  BaseModelAdapter   │  │  Router → Builder por dim.    │
   │  SklearnAdapter     │  │  theme.py (dark mode global)  │
   └─────────────────────┘  └───────────────────────────────┘


Módulo ``api/``
===============

Contiene las funciones de alto nivel que el usuario final invoca directamente.

- ``linear.py`` → :func:`mlektic.api.linear.visualize_lr`
- ``logistic.py`` → :func:`mlektic.api.logistic.visualize_logistic`

Estas funciones orquestan dos pasos:

1. **Captura de historial** (``fit_history`` / ``fit_history_logistic``).
2. **Construcción de la figura** (``build_lr_figure`` / ``build_logistic_figure``).


Módulo ``adapters/``
====================

Implementa el **patrón Adapter** para abstraer la interacción con diferentes frameworks de ML.

- ``BaseModelAdapter`` — Clase abstracta (ABC) que define el contrato: ``predict()``,
  ``predict_proba()``, ``extract_linear_theta()``, ``extract_logistic_theta()``,
  ``fit()``, ``partial_fit()``, ``get_scaler_params()``, ``transform_X()``.
- ``SklearnAdapter`` — Implementación concreta para estimadores y ``Pipeline`` de Scikit-Learn.
  Incluye ``clone_for_replay()`` para crear copias con ``warm_start=True`` y ``max_iter=1``.

.. note::
   El diseño permite añadir adapters para otros frameworks (PyTorch, XGBoost, etc.)
   implementando la interfaz ``BaseModelAdapter``.


Módulo ``domain/``
==================

Define las **estructuras de datos** y contratos internos:

- ``config.py`` — ``LinearHistoryConfig`` y ``LogisticHistoryConfig`` (dataclasses congelados).
- ``history.py`` — ``LinearHistoryPayload`` y ``LogisticHistoryPayload`` (TypedDicts que
  definen el contrato de los diccionarios de historial).


Módulo ``history/``
===================

Corazón del proceso de captura de entrenamiento.

- ``base.py`` — Define ``HistoryCaptureStrategy`` (ABC) y las funciones de conversión
  de parámetros entre espacio escalado y original:
  ``_scale_linear_theta()``, ``_scale_logistic_binary_theta()``,
  ``_scale_logistic_multiclass_theta()``.

- ``engine.py`` — ``HistoryEngine``: fachada que:

  1. Selecciona la estrategia (iterativa vs. interpolación) según el modo.
  2. Aplica suavizado EMA a las series temporales.
  3. Aplica el rescalado de ``θ`` al espacio de visualización solicitado.

- ``strategy_iterative.py`` — ``IterativeCapture``: clona el modelo, configura
  ``warm_start=True`` y ``max_iter=1``, y ejecuta ``partial_fit()`` paso a paso.
  Captura en cada iteración: predicciones, pesos, pérdida.

- ``strategy_interp.py`` — ``InterpolationCapture``: para modelos que no soportan
  entrenamiento iterativo (e.g. ``LinearRegression``). Interpola linealmente entre
  una línea base (media o ceros) y las predicciones finales del modelo.


Módulo ``services/``
====================

Capa de servicio que conecta la API pública con el motor de historial:

- ``linear_history.py`` — ``fit_history()`` y ``fit_history_logistic()`` instancian
  la configuración, crean el ``HistoryEngine`` y ejecutan la captura.
- ``logistic_history.py`` — Re-exporta ``fit_history_logistic`` para mantener
  compatibilidad de imports.


Módulo ``utils/``
=================

Funciones matemáticas y de utilidad puras:

- ``math.py`` — ``_sigmoid()``, ``_softmax()``, ``_binary_log_loss_from_p()``,
  ``_multiclass_cross_entropy()``, ``_one_hot()``, ``_ema_smooth()``.
- ``grids.py`` — ``build_1d_grid()`` y ``build_2d_grid()`` para crear los meshgrids
  sobre los cuales se evalúan las predicciones.


Módulo ``visualization/``
=========================

Responsable de traducir el historial capturado en figuras Plotly animadas.

- ``theme.py`` — Funciones de tema global: ``get_base_layout()`` (dark mode),
  ``get_updatemenus()`` (botones Play/Pause), ``get_sliders()``,
  ``get_legend_props()``, ``create_annotation()``.

- **``linear/``**:

  - ``router.py`` → ``build_lr_figure()``: detecta la dimensionalidad y delega.
  - ``simple.py`` → 1 variable: scatter 2D + recta animada + curva MSE.
  - ``plane.py`` → 2 variables: scatter 3D + plano animado + curva MSE.
  - ``multivar.py`` → d > 2: matriz LaTeX con ``θ`` actualizado por frame + curva MSE.

- **``logistic/``**:

  - ``router.py`` → ``build_logistic_figure()``: detecta dimensión y número de clases.
  - ``binary_1d.py`` → Clasificación binaria 1D: curva sigmoide animada.
  - ``binary_2d.py`` → Clasificación binaria 2D: superficie de probabilidad 3D.
  - ``binary_nd.py`` → Clasificación binaria d > 2: matriz LaTeX de pesos.
  - ``multiclass_1d.py`` → Multiclase 1D: curvas de probabilidad por clase.
  - ``multiclass_2d.py`` → Multiclase 2D: superficies Softmax superpuestas.
  - ``multiclass_nd.py`` → Multiclase d > 2: matriz de pesos multiclase.


Escalabilidad hacia Nuevos Modelos
===================================

La frontera de extensión principal es ``BaseModelAdapter``. La API pública y
los builders de visualización no dependen directamente de Scikit-Learn; dependen
del contrato del adapter y del payload de historial.

Para añadir nuevas familias de modelos, por ejemplo PyTorch, Keras, XGBoost o
capas de redes neuronales artificiales, el nuevo adapter debe implementar:

- ``predict()`` y, para clasificación, ``predict_proba()``.
- Extracción de parámetros cuando exista una forma interpretable para la figura.
- ``fit()`` / ``partial_fit()`` o una estrategia de replay equivalente.
- Transformación de features y metadatos de escalado cuando el modelo use
  preprocesamiento externo.

El ``HistoryEngine`` orquesta captura, métricas, suavizado, rescalado y
decimación temporal. Las estrategias de captura producen historiales con una
forma común, y los routers de visualización eligen el builder por dimensión y
tipo de tarea. Este diseño permite sumar nuevos modelos sin duplicar la lógica
de métricas ni las figuras por dimensión.


Módulo ``_internal/``
=====================

Helpers heredados del código original, mantenidos por compatibilidad:

- ``common.py`` — Funciones como ``_as_2d()``, ``_as_1d()``, ``_get_final_estimator()``,
  ``_find_standard_scaler()``, ``_make_iterative_replay_estimator()``, etc.
  Muchas de estas funciones están duplicadas en los módulos refactorizados
  (``adapters/``, ``utils/``), pero se mantienen para no romper imports internos.
