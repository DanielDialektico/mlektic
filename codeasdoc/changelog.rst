=================
Changelog
=================

v0.1.0 (En desarrollo)
========================

Primera versión de la librería.

Funcionalidades
----------------

- **Redes Neuronales PyTorch**: soporte opcional mediante ``mlektic[torch]`` con
  grafo temporal de pesos como mapa de calor, gradientes de backpropagation
  simultaneos, arquitectura matematica, dimensiones y formulas LaTeX, loss y tres
  metricas independientes, evolucion matricial de parametros, explicacion temporal
  del forward pass y reportes HTML completos para redes grandes.

- **Regresión Lineal**: Soporte completo para ``visualize_lr()`` con:
  - Renderizado 2D (1 variable): recta de regresión animada + curva MSE.
  - Renderizado 3D (2 variables): plano predictivo animado.
  - Renderizado LaTeX (d > 2): matriz de parámetros interactiva.
  - **NUEVO**: ``explain_lr_prediction()`` para explicar matemáticamente y de forma visual las predicciones de un modelo ya entrenado.

- **Regresión Logística**: Soporte completo para ``visualize_logistic()`` con:
  - Clasificación binaria: 1D (sigmoide), 2D (superficie), d > 2 (LaTeX).
  - Clasificación multiclase: 1D (curvas de probabilidad), d > 2 (matrices de pesos).

- **Métricas Dinámicas**: Parámetro ``metrics`` para mostrar simultáneamente variables como ``loss``, ``mse``, ``r2``, y formateo inteligente de números para evitar desbordes visuales.
- **Métricas Reutilizables**: Builders internos para métricas por frame en regresión lineal y logística, incluyendo ``loss``, ``mse``, ``r2``, ``mae``, ``accuracy`` y ``f1``, además de métricas personalizadas.
- **Muestreo de Historiales**: Utilidades internas para reducir historiales largos mediante ``max_frames`` o ``frame_step`` antes del renderizado.

- **Integración con Scikit-Learn**: Compatible con estimadores directos y ``Pipeline``.

- **Adapter Pattern**: ``SklearnAdapter`` como implementación concreta del
  ``BaseModelAdapter`` (extensible a otros frameworks).
- **Ruta de Escalabilidad**: Documentación del contrato de adapters para futuras familias de modelos, incluyendo frameworks no Scikit-Learn y visualizaciones de redes neuronales.

- **Strategy Pattern**: Dos estrategias de captura de historial:
  - ``IterativeCapture`` para modelos iterativos (``partial_fit`` / ``warm_start``).
  - ``InterpolationCapture`` para modelos no iterativos.

- **Suavizado EMA**: Opción de suavizar curvas de pérdida y predicciones.

- **Espacio de visualización**: Soporte para ``display_space="original"`` y ``"scaled"``
  con transformación inversa automática de parámetros escalados.

- **Tema visual unificado**: Dark mode con Plotly, botones Play/Pause, slider temporal.

Fixes y Mejoras Recientes
-------------------------

- **Estabilidad en Exportación HTML**: Corrección del "glitch" de redimensionamiento de arrays de JavaScript en Plotly; los arrays ahora se rellenan con ``None`` para garantizar un trazado de línea constante y evitar que las animaciones HTML se corten.
- **Formateo Multivariable**: La función ``explain_lr_prediction()`` para 3 o más variables ahora formatea correctamente la coordenada resultante incluyendo ``y_hat`` al final (ej., ``(x_1, \dots, x_d, \hat{y})``), logrando consistencia matemática con las vistas de 1D y 2D.
- **Exportaciones Públicas**: ``explain_lr_prediction`` ahora se importa correctamente desde ``mlektic``.
- **Métricas Logísticas**: Las métricas de clasificación respetan las etiquetas reales de clase en vez de asumir índices ``0..K-1``.
- **Pipelines en Interpolación**: Los historiales interpolados conservan metadatos de escalado para evaluar métricas en el espacio visual solicitado.
