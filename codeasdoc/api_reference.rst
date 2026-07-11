=======================
Referencia de la API
=======================

Esta sección documenta de forma exhaustiva todas las funciones públicas exportadas
por la librería ``mlektic``.


API de Alto Nivel
==================

.. module:: mlektic.api.linear
   :synopsis: API pública para regresión lineal.

.. autofunction:: mlektic.api.linear.visualize_lr

.. autofunction:: mlektic.api.linear.explain_lr_prediction


.. module:: mlektic.api.logistic
   :synopsis: API pública para regresión logística.

.. autofunction:: mlektic.api.logistic.visualize_logistic

.. autofunction:: mlektic.api.logistic.explain_logistic_prediction


Servicios de Historial
=======================

.. module:: mlektic.services.linear_history
   :synopsis: Funciones de captura de historial.

.. autofunction:: mlektic.services.linear_history.fit_history

.. autofunction:: mlektic.services.linear_history.fit_history_logistic


Motor de Historial
===================

.. module:: mlektic.history.engine
   :synopsis: Motor de captura de historial.

.. autoclass:: mlektic.history.engine.HistoryEngine
   :members:
   :undoc-members:
   :show-inheritance:


Estrategias de Captura
=======================

.. module:: mlektic.history.base
   :synopsis: Interfaz base de estrategia.

.. autoclass:: mlektic.history.base.HistoryCaptureStrategy
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: mlektic.history.base._scale_linear_theta

.. autofunction:: mlektic.history.base._scale_logistic_binary_theta

.. autofunction:: mlektic.history.base._scale_logistic_multiclass_theta


.. module:: mlektic.history.strategy_iterative
   :synopsis: Estrategia iterativa.

.. autoclass:: mlektic.history.strategy_iterative.IterativeCapture
   :members:
   :undoc-members:
   :show-inheritance:


.. module:: mlektic.history.strategy_interp
   :synopsis: Estrategia de interpolación.

.. autoclass:: mlektic.history.strategy_interp.InterpolationCapture
   :members:
   :undoc-members:
   :show-inheritance:


Adapters
========

.. module:: mlektic.adapters.base
   :synopsis: Clase base para adapters de modelos.

.. autoclass:: mlektic.adapters.base.BaseModelAdapter
   :members:
   :undoc-members:
   :show-inheritance:


.. module:: mlektic.adapters.sklearn
   :synopsis: Adapter para Scikit-Learn.

.. autoclass:: mlektic.adapters.sklearn.SklearnAdapter
   :members:
   :undoc-members:
   :show-inheritance:


Dominio
=======

.. module:: mlektic.domain.config
   :synopsis: Configuración de captura.

.. autoclass:: mlektic.domain.config.LinearHistoryConfig
   :members:
   :undoc-members:

.. autoclass:: mlektic.domain.config.LogisticHistoryConfig
   :members:
   :undoc-members:


.. module:: mlektic.domain.history
   :synopsis: Contratos de payload.

.. autoclass:: mlektic.domain.history.LinearHistoryPayload
   :members:
   :undoc-members:

.. autoclass:: mlektic.domain.history.LogisticHistoryPayload
   :members:
   :undoc-members:


Routers de Visualización
=========================

.. module:: mlektic.visualization.linear.router
   :synopsis: Enrutador para figuras de regresión lineal.

.. autofunction:: mlektic.visualization.linear.router.build_lr_figure


.. module:: mlektic.visualization.logistic.router
   :synopsis: Enrutador para figuras de regresión logística.

.. autofunction:: mlektic.visualization.logistic.router.build_logistic_figure


Builders de Figuras — Regresión Lineal
=======================================

.. module:: mlektic.visualization.linear.simple
   :synopsis: Figura de regresión lineal simple (1 variable).

.. autofunction:: mlektic.visualization.linear.simple.build_simple_lr_figure


.. module:: mlektic.visualization.linear.plane
   :synopsis: Figura de regresión lineal con plano (2 variables).

.. autofunction:: mlektic.visualization.linear.plane.build_plane_lr_figure


.. module:: mlektic.visualization.linear.multivar
   :synopsis: Figura de regresión lineal multivariable (d > 2).

.. autofunction:: mlektic.visualization.linear.multivar.build_multivar_lr_figure


Builders de Figuras — Regresión Logística
==========================================

.. module:: mlektic.visualization.logistic.binary_1d
   :synopsis: Figura de regresión logística binaria (1 variable).

.. autofunction:: mlektic.visualization.logistic.binary_1d.build_binary_simple_logistic_figure


.. module:: mlektic.visualization.logistic.binary_2d
   :synopsis: Figura de regresión logística binaria (2 variables).

.. autofunction:: mlektic.visualization.logistic.binary_2d.build_binary_plane_logistic_figure


.. module:: mlektic.visualization.logistic.binary_nd
   :synopsis: Figura de regresión logística binaria (d > 2).

.. autofunction:: mlektic.visualization.logistic.binary_nd.build_binary_multivar_logistic_figure


.. module:: mlektic.visualization.logistic.multiclass_1d
   :synopsis: Figura multiclase (1 variable).

.. autofunction:: mlektic.visualization.logistic.multiclass_1d.build_multiclass_1d_logistic_figure


.. module:: mlektic.visualization.logistic.multiclass_2d
   :synopsis: Figura multiclase (2 variables).

.. autofunction:: mlektic.visualization.logistic.multiclass_2d.build_multiclass_2d_logistic_figure


.. module:: mlektic.visualization.logistic.multiclass_nd
   :synopsis: Figura multiclase (d > 2).

.. autofunction:: mlektic.visualization.logistic.multiclass_nd.build_multiclass_multivar_logistic_figure


Utilidades Matemáticas
=======================

.. module:: mlektic.utils.math
   :synopsis: Funciones matemáticas auxiliares.

.. autofunction:: mlektic.utils.math._sigmoid

.. autofunction:: mlektic.utils.math._softmax

.. autofunction:: mlektic.utils.math._binary_log_loss_from_p

.. autofunction:: mlektic.utils.math._multiclass_cross_entropy

.. autofunction:: mlektic.utils.math._one_hot

.. autofunction:: mlektic.utils.math._ema_smooth


Utilidades de Grids
====================

.. module:: mlektic.utils.grids
   :synopsis: Generación de meshgrids.

.. autofunction:: mlektic.utils.grids.build_1d_grid

.. autofunction:: mlektic.utils.grids.build_2d_grid


Métricas e Historial
====================

.. module:: mlektic.history.metrics
   :synopsis: Cálculo de métricas por frame.

.. autofunction:: mlektic.history.metrics.build_linear_metrics

.. autofunction:: mlektic.history.metrics.build_logistic_metrics


.. module:: mlektic.history.sampling
   :synopsis: Muestreo temporal de historiales.

.. autofunction:: mlektic.history.sampling.decimate_history


Tema Visual
============

.. module:: mlektic.visualization.theme
   :synopsis: Funciones de tema y layout para Plotly.

.. autofunction:: mlektic.visualization.theme.get_base_layout

.. autofunction:: mlektic.visualization.theme.get_legend_props

.. autofunction:: mlektic.visualization.theme.get_updatemenus

.. autofunction:: mlektic.visualization.theme.get_sliders

.. autofunction:: mlektic.visualization.theme.create_annotation
