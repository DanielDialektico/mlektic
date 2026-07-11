# Mlektic

**Mlektic** es una librería de Python diseñada para demostrar visual y matemáticamente cómo evolucionan los modelos de *Machine Learning* durante su fase de entrenamiento. Provee gráficos y animaciones interactivas impulsadas por `plotly`, creadas específicamente para entender las tripas de los algoritmos de Scikit-Learn.

---

## 🚀 Características Principales

*   **Integración Nivel-Cero con Scikit-Learn**: Compatible directamente con estimadores iterativos (como `SGDRegressor`, `SGDClassifier`) y `Pipelines` estándar.
*   **Animaciones Fluidas**: Visualiza en tiempo real cómo los parámetros (`θ`), la recta/curva de predicción y la función de pérdida (Loss) convergen.
*   **Regresión Lineal y Logística**: Soporte completo para los dos tipos de regresión más fundamentales del ML, cada uno con su propia función pública.
*   **Renderizado Inteligente por Dimensión**:
    *   **1 Variable (2D)**: Dibuja la recta de regresión / curva sigmoide ajustándose punto a punto junto a la curva de pérdida.
    *   **2 Variables (3D)**: Renderiza un plano predictivo / superficie de probabilidad en 3D interactivo que se inclina y eleva iteración por iteración. En el caso de logística multiclase, renderiza $K$ superficies superpuestas evaluando el clasificador Softmax de forma dinámica sobre los datos.
    *   **Múltiples Variables (d > 2)**: Al no ser posible graficar predicciones de alta dimensión, `mlektic` construye dinámicamente una matriz matemática en LaTeX interactiva que actualiza los pesos de tu vector `θ` en tiempo real.
*   **Clasificación Multiclase**: Visualización automática de curvas de probabilidad por clase (1D), matrices de pesos multiclase (d > 2), y superficies múltiples simultáneas con paneles de ecuaciones (2D).
*   **Inspección de Pipelines**: Capacidad de proyectar el aprendizaje visualmente tanto en el **"espacio local/escalado"** como de vuelta al **"espacio original"** cuando usas funciones como `StandardScaler`.

---

## 📦 Instalación y Configuración

El proyecto está diseñado usando las mejores prácticas de Python modernas (PEP 621) y `uv` como gestor ultrarrápido de dependencias.

Para desarrollar o instalar localmente:
```bash
git clone https://github.com/DanielDialektico/mlektic.git
cd mlektic

# Crear entorno virtual e instalar dependencias 
uv sync
```

---

## 💡 Quickstart — Regresión Lineal

La API pública para regresión lineal se resume en `visualize_lr`. Todo el trazado dimensional es manejado de manera automática.

```python
import numpy as np
import plotly.io as pio
from sklearn.linear_model import SGDRegressor
from mlektic import visualize_lr

# Fuerza el renderizador incrustado si usas VS Code Jupyter
pio.renderers.default = "notebook" 

# 1. Generar datos de juguete
X = np.sort(np.random.rand(100, 1)) * 10
y = 2.5 * X.ravel() + 1.0 + np.random.randn(100) * 2

# 2. Tu modelo de Scikit-Learn
model = SGDRegressor(
    loss="squared_error",
    max_iter=50,
    learning_rate="constant",
    eta0=0.005,
    random_state=42
)
model.fit(X, y)

# 3. Animar la magia
fig = visualize_lr(
    model, X, y,
    steps=60,               # Cantidad de cuadros (frames) a simular
    frame_duration=80,      # Velocidad (ms por frame). Ej: 10 para super rápido
    show_loss=True,         # Muestra la curva MSE/Loss al costado
    title="Mi Primera Animación Mlektic"
)
fig.show()

# (Opcional) Si la animación en tu editor es lenta, expórtalo a HTML puro:
# fig.write_html("animacion.html", auto_play=False) 
```

---

## 💡 Quickstart — Regresión Logística

La API pública para regresión logística es `visualize_logistic`. Soporta clasificación binaria y multiclase.

```python
import numpy as np
import plotly.io as pio
from sklearn.linear_model import SGDClassifier
from mlektic import visualize_logistic

pio.renderers.default = "notebook"

# 1. Generar datos de clasificación binaria
np.random.seed(42)
n = 200
X = np.random.randn(n, 1)
y = (X.ravel() > 0).astype(int)

# 2. Tu clasificador de Scikit-Learn
model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.05,
    max_iter=500,
    random_state=42
)
model.fit(X, y)

# 3. Animar la curva sigmoide convergiendo
fig = visualize_logistic(
    model, X, y,
    steps=60,
    show_loss=True,
    frame_duration=80,
    title="Regresión Logística Binaria"
)
fig.show()
```

---

## 🛠 Opciones Avanzadas de Visualización

### Parámetros de `visualize_lr`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `trained_estimator` | estimator / Pipeline | — | Modelo de Scikit-Learn ya entrenado. |
| `X` | `np.ndarray` | — | Matriz de features de entrenamiento. |
| `y` | `np.ndarray` | — | Vector objetivo. |
| `steps` | `int` | `60` | Número de frames de animación. |
| `mode` | `str` | `"auto"` | Estrategia de captura: `"auto"`, `"iterative"` o `"final_interp"`. |
| `show_loss` | `bool` | `True` | Muestra la curva de pérdida junto al gráfico principal. |
| `title` | `str` | `None` | Título personalizado del gráfico. |
| `smooth` | `str \| None` | `"ema"` | Suavizado de la curva de pérdida (`"ema"` o `None`). |
| `smooth_beta` | `float` | `0.85` | Parámetro beta para el suavizado EMA. |
| `strict_loss` | `bool` | `False` | Si `True`, lanza error si el loss no se puede animar correctamente. |
| `baseline` | `str` | `"mean"` | Referencia inicial del gráfico de pérdida (`"mean"` o `"zeros"`). |
| `display_space` | `str` | `"original"` | Espacio de visualización de parámetros (`"original"` o `"scaled"`). |
| `metrics` | `list[str]` | `["loss", "mse", "r2"]` | Lista de métricas a calcular y mostrar ("loss", "mse", "r2", "mae"). |
| `dec` | `int` | `4` | Decimales para formatear los parámetros. |
| `frame_duration` | `int` | `80` | Duración de cada frame en ms. Disminuir para más velocidad. |

### Parámetros de `visualize_logistic`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `trained_estimator` | estimator / Pipeline | — | Clasificador de Scikit-Learn ya entrenado. |
| `X` | `np.ndarray` | — | Matriz de features de entrenamiento. |
| `y` | `np.ndarray` | — | Vector de etiquetas. |
| `steps` | `int` | `60` | Número de frames de animación. |
| `mode` | `str` | `"auto"` | Estrategia de captura: `"auto"`, `"iterative"` o `"final_interp"`. |
| `show_loss` | `bool` | `True` | Muestra la curva de log-loss junto al gráfico principal. |
| `title` | `str` | `None` | Título personalizado del gráfico. |
| `smooth` | `str \| None` | `"ema"` | Suavizado de la curva de pérdida (`"ema"` o `None`). |
| `smooth_beta` | `float` | `0.85` | Parámetro beta para el suavizado EMA. |
| `strict_loss` | `bool` | `False` | Si `True`, lanza error si el loss no se puede animar correctamente. |
| `baseline` | `str` | `"prior"` | Referencia inicial del loss: `"prior"` (proporciones de clase) o `"uniform"`. |
| `display_space` | `str` | `"original"` | Espacio de visualización de parámetros (`"original"` o `"scaled"`). |
| `metrics` | `list[str]` | `["loss", "accuracy"]` | Lista de métricas a mostrar durante el entrenamiento. |
| `dec` | `int` | `4` | Decimales para formatear los parámetros. |
| `frame_duration` | `int` | `80` | Duración de cada frame en ms. |

---

## 🔍 Explicación Visual de Predicciones (`explain_lr_prediction`, `explain_logistic_prediction`)

`mlektic` incluye herramientas diseñadas para explicar de forma matemática y geométrica una predicción puntual de tu modelo ya entrenado. Soporta Scikit-Learn pipelines y formatea inteligentemente los pesos.

```python
from mlektic.api.linear import explain_lr_prediction
from mlektic.api.logistic import explain_logistic_prediction

# 1. Escoge un punto de prueba (forma 2D)
x_query = np.array([[150.0, 25.0]])

# 2. Haz la predicción con tu modelo lineal o logístico
yhat = model.predict(x_query)[0]

# 3. Explica visualmente de dónde salió el valor
fig = explain_lr_prediction( # o explain_logistic_prediction
    model, X_train, y_train,
    x_query=x_query,
    yhat=yhat,
    display_space="original" # Permite ver cómo opera en espacio escalado o nativo
)
fig.show()
```

---

## 🏗 Arquitectura del Proyecto

```
src/mlektic/
├── __init__.py              # Exportaciones públicas
├── core.py                  # Fachada para regresión lineal
├── logistic.py              # Fachada para regresión logística
├── api/
│   ├── linear.py            # API pública: visualize_lr()
│   └── logistic.py          # API pública: visualize_logistic()
├── adapters/
│   ├── base.py              # BaseModelAdapter (ABC)
│   └── sklearn.py           # SklearnAdapter (Scikit-Learn)
├── domain/
│   ├── config.py            # LinearHistoryConfig, LogisticHistoryConfig
│   └── history.py           # TypedDicts: LinearHistoryPayload, LogisticHistoryPayload
├── history/
│   ├── base.py              # HistoryCaptureStrategy (ABC) + funciones de rescalado θ
│   ├── engine.py            # HistoryEngine: orquesta captura + suavizado + rescalado
│   ├── strategy_interp.py   # InterpolationCapture (modelos no iterativos)
│   └── strategy_iterative.py# IterativeCapture (modelos con partial_fit/warm_start)
├── services/
│   ├── linear_history.py    # fit_history() y fit_history_logistic()
│   └── logistic_history.py  # Re-export de fit_history_logistic
├── utils/
│   ├── math.py              # sigmoid, softmax, log-loss, EMA, one-hot
│   └── grids.py             # Generación de meshgrids 1D y 2D
├── visualization/
│   ├── theme.py             # Tema visual (dark mode, sliders, play/pause)
│   ├── linear/
│   │   ├── router.py        # build_lr_figure(): enruta por dimensión
│   │   ├── simple.py        # 1 variable (recta + scatter 2D)
│   │   ├── plane.py         # 2 variables (plano 3D)
│   │   └── multivar.py      # d > 2 (matriz LaTeX interactiva)
│   └── logistic/
│       ├── router.py        # build_logistic_figure(): enruta por dimensión y clases
│       ├── binary_1d.py     # Binaria, 1 variable (curva sigmoide)
│       ├── binary_2d.py     # Binaria, 2 variables (superficie 3D)
│       ├── binary_nd.py     # Binaria, d > 2 (matriz LaTeX)
│       ├── multiclass_1d.py # Multiclase, 1 variable (curvas de probabilidad)
│       └── multiclass_nd.py # Multiclase, d > 2 (matriz de pesos)
└── _internal/
    └── common.py            # Helpers compartidos (legacy/compatibilidad)
```

---

## 📐 Funciones de Bajo Nivel

Además de la API de alto nivel (`visualize_lr`, `visualize_logistic`), puedes usar las funciones granulares:

### Regresión Lineal
*   `fit_history(estimator, X, y, ...)` → Captura el historial de entrenamiento como un diccionario.
*   `build_lr_figure(X, y, history=...)` → Construye la figura Plotly a partir del historial.
*   `build_simple_lr_figure(...)` → Figura para 1 variable.
*   `build_plane_lr_figure(...)` → Figura para 2 variables.
*   `build_multivar_lr_figure(...)` → Figura para d > 2 variables.

### Regresión Logística
*   `fit_history_logistic(estimator, X, y, ...)` → Captura el historial logístico.
*   `build_logistic_figure(X, y, history=...)` → Construye la figura Plotly logística.
*   `build_binary_simple_logistic_figure(...)` → Figura binaria 1D.
*   `build_binary_plane_logistic_figure(...)` → Figura binaria 2D.
*   `build_binary_multivar_logistic_figure(...)` → Figura binaria d > 2.
*   `build_multiclass_1d_logistic_figure(...)` → Figura multiclase 1D.
*   `build_multiclass_multivar_logistic_figure(...)` → Figura multiclase d > 2.

---

## 🧪 Directorio Local de Pruebas

Si acabas de clonar el repositorio, puedes probar todas las capacidades multi-dimensionales sin tener que escribir código de prueba verificando el directorio pre-empaquetado `/local_test`. Adentro encontrarás:

### Regresión Lineal
- `lg_test_1_var.py` — 1 variable con diferentes escenarios de pipelines y sin escalar.
- `lg_test_plane.py` — 2 variables.
- `lg_test_multivar_pipeline.py` — Pruebas con regresión de d=2 hasta d=30 en formato matricial iterativo y estático.
- `tg_pred_test_1.py` / `tg_pred_test_2.py` — Casos de uso de la herramienta explicativa (`explain_lr_prediction`).

### Regresión Logística
- `test_log_var.py` — Clasificación binaria con datos reales (Breast Cancer) y sintéticos, con y sin escalado.

---

## 📚 Documentación

La documentación técnica completa generada con Sphinx se encuentra en el directorio `codeasdoc/`. Para compilarla localmente:

```bash
cd codeasdoc
pip install sphinx sphinx-rtd-theme
make html          # Linux/macOS
.\make.bat html    # Windows
```

La documentación compilada se encontrará en `codeasdoc/_build/html/index.html`.

---

## 🤝 Contribuciones

Si formas parte del equipo de desarrollo, te pedimos ejecutar la suite de linter y testeo antes de cada *Commit*:
```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

---

## 📄 Licencia

Este proyecto se distribuye bajo los términos descritos en el repositorio. Consulta el archivo correspondiente para más detalles.
