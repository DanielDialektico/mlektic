=================
Changelog
=================

v0.1.0 (En desarrollo)
========================

Primera versión de la librería.

Funcionalidades
----------------

- **Redes Neuronales PyTorch**: soporte opcional mediante ``mlektic[torch]`` con
  grafo temporal de pesos y salidas neuronales como mapas de calor independientes,
  escalas globales reales por defecto, normalizacion temporal opcional por capa y
  modo adicional para colorear aristas mediante la senal :math:`\theta_{ji}a_i`,
  gradientes de backpropagation simultaneos, arquitectura matematica, dimensiones y
  formulas LaTeX, cuadricula 2 por 2 para loss y tres metricas inferibles desde
  predicciones y objetivos, evolucion matricial de
  parametros, explicacion temporal del forward pass y reportes HTML completos para
  redes grandes.
  La lectura compacta de pesos y el paso temporal se actualizan como trazas animadas
  sin redibujar toda la red, y el titulo conserva separacion respecto a la ecuacion.
  Los ceros exactos de ReLU se identifican en el hover; las etiquetas temporales y
  escalas matematicas conservan margen respecto a los bordes de la figura.

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
- **Ruta de Escalabilidad**: contrato de adapters para futuras familias tabulares
  y contrato especializado de historiales para PyTorch y otros frameworks neurales.

- **Strategy Pattern**: Dos estrategias de captura de historial:
  - ``IterativeCapture`` para modelos iterativos (``partial_fit`` / ``warm_start``).
  - ``InterpolationCapture`` para modelos no iterativos.

- **Suavizado EMA**: Opción de suavizar la curva de pérdida sin modificar la
  geometría ni los parámetros exactos del modelo.

- **Espacio de visualización**: Soporte para ``display_space="original"`` y ``"scaled"``
  con transformación inversa automática de parámetros escalados.

- **Tema visual unificado**: Dark mode con Plotly, botones Play/Pause, slider temporal.

Fixes y Mejoras Recientes
-------------------------

- **Contrato de Animacion Documentado**: se distingue entre LaTeX simbolico fijo,
  sustituciones MathJax nativas y trazas numericas hibridas; tambien se explica la
  division de ``frame_duration`` entre subframes y la recomendacion de 30 a 45 FPS
  para Jupyter y Colab.
- **Controles Estables**: Play y Pause permanecen blancos con texto negro en
  todos sus estados, incluso cuando Plotly reconstruye superficies 3D, sin
  seguimiento JavaScript ni cambios en la interpolación híbrida.
- **Métricas Laterales**: la vista híbrida 1D apila sus recuadros de métricas en
  una columna independiente para evitar superposiciones.
- **Predicciones sin Desbordes**: las sustituciones 2D usan notación científica
  LaTeX y tipografía compacta; las vistas ND muestran vectores columna truncados
  con ``\vdots`` en lugar de cuadrículas que pudieran confundirse con matrices.
- **Notacion Unificada**: regresion lineal, logistica y redes neuronales usan
  :math:`\theta`, :math:`\Theta` y :math:`\boldsymbol{\theta}_0` con dimensiones
  independientes del acomodo visual de las matrices.
- **Enlace Multiclase Riguroso**: deteccion automatica de Softmax frente a
  sigmoides OvR normalizadas, aplicada a historiales, figuras y sustituciones.
- **Transiciones Suaves**: interpolacion configurable para trazas 2D mediante
  ``transition_duration``, con margen entre frames para evitar rectas parcialmente
  dibujadas, rutas SVG estables sin simplificacion variable y redibujado estable
  para superficies 3D.
- **LaTeX Animado en Jupyter**: los frames que modifican ecuaciones o metricas en
  el layout activan redibujado selectivo para que los valores de :math:`\theta`
  evolucionen junto con el slider; el orden ``traces first`` preserva a la vez la
  interpolacion suave de rectas y curvas 2D.
- **Animacion Hibrida 1D**: recta, coeficientes numericos, perdida y metricas
  comparten subframes sincronizados sin redibujar el layout; el slider conserva
  exclusivamente checkpoints semanticos y la definicion simbolica sigue en LaTeX.
- **EMA Matematicamente Consistente**: ``smooth="ema"`` suaviza solamente la
  perdida y nunca modifica geometria, probabilidades o parametros del modelo.
- **Layouts de Alta Dimension**: normalizadores compactos y paneles ND sin
  columnas vacias ni formulas que invadan graficas adyacentes.
- **Documentacion Neural Completa**: README, inicio rapido, arquitectura, modos
  del grafo, escalas globales exactas, metricas automaticas, ceros de ReLU,
  trazas animadas de parametros y visualizacion/exportacion de reportes HTML
  quedaron alineados con la API publica y sus docstrings.
- **Estabilidad en Exportación HTML**: Corrección del "glitch" de redimensionamiento de arrays de JavaScript en Plotly; los arrays ahora se rellenan con ``None`` para garantizar un trazado de línea constante y evitar que las animaciones HTML se corten.
- **Formateo Multivariable**: La función ``explain_lr_prediction()`` para 3 o más variables ahora formatea correctamente la coordenada resultante incluyendo ``y_hat`` al final (ej., ``(x_1, \dots, x_d, \hat{y})``), logrando consistencia matemática con las vistas de 1D y 2D.
- **Exportaciones Públicas**: ``explain_lr_prediction`` ahora se importa correctamente desde ``mlektic``.
- **Métricas Logísticas**: Las métricas de clasificación respetan las etiquetas reales de clase en vez de asumir índices ``0..K-1``.
- **Pipelines en Interpolación**: Los historiales interpolados conservan metadatos de escalado para evaluar métricas en el espacio visual solicitado.
