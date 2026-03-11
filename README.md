# Mlektic

**Mlektic** es una librería de Python diseñada para demostrar visual y matemáticamente cómo evolucionan los modelos de *Machine Learning* durante su fase de entrenamiento. Provee gráficos y animaciones interactivas impulsadas por `plotly`, creadas específicamente para entender las tripas de los algoritmos de Scikit-Learn.

---

## 🚀 Características Principales

*   **Integración Nivel-Cero con Scikit-Learn**: Compatible directamente con estimadores iterativos (como `SGDRegressor`) y `Pipelines` estándar.
*   **Animaciones Fluidas**: Visualiza en tiempo real cómo los parámetros (`θ`), la recta de predicción y la función de pérdida (Loss) convergen.
*   **Renderizado Inteligente por Dimensión**:
    *   **1 Variable (2D)**: Dibuja la recta de regresión ajustándose punto a punto junto a la curva de pérdida.
    *   **2 Variables (3D)**: Renderiza un plano predictivo en 3D interactivo que se inclina y eleva iteración por iteración.
    *   **Múltiples Variables (d > 2)**: Al no ser posible graficar predicciones de alta dimensión, `mlektic` construye dinámicamente una matriz matemática en LaTeX interactiva que actualiza los pesos de tu vector `θ` en tiempo real.
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

## 💡 Quickstart (Uso Básico)

La API pública está simplificada a una sola función: `visualize_lr`. Todo el trazado dimensional es manejado de manera automática.

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

## 🛠 Opciones Avanzadas de Visualización

La función `visualize_lr` cuenta con opciones de fine-tuning visual (kwargs):

- **`frame_duration`** *(int)*: Controla la velocidad de la animación. El valor por defecto es `80`ms. Disminuir este valor (ej. a `10`) acelerará el renderizado.
- **`display_space`** *(str)*: `["original", "scaled"]`. Si entrenaste tu modelo usando un `Pipeline` con `StandardScaler`, puedes pedirle a la herramienta que destransforme y anime los pesos `θ` vistos desde el espacio original de los datos en lugar de tus variables estandarizadas.
- **`smooth`** *(str o None)*: Si los gradientes de tu modelo rebotan demasiado (común en Stochastic Gradient Descent), seteado en `"ema"` suavizará la curva de pérdida. Si prefieres la pérdida cruda paso-a-paso, ponlo en `"none"`.
- **`baseline`** *(str)*: Dónde fijar el borde superior o inicial del gráfico de la pérdida. Puede ser la media del vector y (`"mean"`) o desde cero absoluto (`"zeros"`).

---

## 🧪 Directorio Local de Pruebas

Si acabas de clonar el repositorio, puedes probar todas las capacidades multi-dimensionales sin tener que escribir código de prueba verificando el directorio pre-empaquetado `/local_test`. Adentro encontrarás:
- `test_1_var.py`
- `test_2_vars.py`
- `test_multivar_pipeline.py` (Incluyendo ejemplos extremos simulados de 100 y 150 dimensiones).

---

## 🤝 Contribuciones

Si formas parte del equipo de desarrollo, te pedimos ejecutar la suite de linter y testeo antes de cada *Commit*:
```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```
