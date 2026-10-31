"""Generate Mlektic's deterministic learning and human visual-QA notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
QA_SIZE_PRESETS = ("default", "compact", "notebook", "wide", "classroom")


def markdown(text: str):
    """Create a Markdown cell."""
    source = text.strip()
    return nbf.v4.new_markdown_cell(source, id=_cell_id("markdown", source))


def code(source: str, *, case_id: str | None = None, tags: list[str] | None = None):
    """Create a cleared code cell with stable project metadata."""
    metadata = {}
    if case_id:
        metadata["mlektic_case_id"] = case_id
    if tags:
        metadata["tags"] = tags
    source = source.strip()
    stable_key = case_id or source
    return nbf.v4.new_code_cell(
        source,
        metadata=metadata,
        execution_count=None,
        outputs=[],
        id=_cell_id("code", stable_key),
    )


def _cell_id(kind: str, value: str) -> str:
    """Return a reproducible Jupyter cell identifier."""
    return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:12]


def setup(imports: str) -> list:
    """Return the standard reproducible notebook preamble."""
    return [
        markdown(
            """
> Run this notebook from top to bottom after installing the project with
> `pip install -e \".[notebooks]\"`. Figures are genuine public-API outputs.
> Cells intentionally contain no assertions: automated invariants live in
> `tests/`, while this notebook is for human visual inspection.
"""
        ),
        code(
            f"""
from pathlib import Path
import sys

candidate = Path.cwd().resolve()
while candidate != candidate.parent and not (candidate / "pyproject.toml").exists():
    candidate = candidate.parent
if not (candidate / "pyproject.toml").exists():
    raise RuntimeError("Open this notebook from inside the Mlektic repository.")
ROOT = candidate
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

{imports}
"""
        ),
    ]


def case(case_id: str, purpose: str, source: str):
    """Create an explained, separately executable visual-QA case."""
    return [
        markdown(f"### `{case_id}`\n\n**Inspect:** {purpose}"),
        code(f'case_heading("{case_id}", "{purpose}")\n{source}', case_id=case_id),
    ]


def add_size_coverage(cells: list, prefix: str, source_template: str) -> None:
    """Append one representative visual case for every public canvas size."""
    cells.append(
        markdown(
            """
## Canvas-size coverage

Run the same representative figure in every supported size. Inspect clipping,
mathematical spacing, controls, legends, axes, and any below-plot derivation.
"""
        )
    )
    for size in QA_SIZE_PRESETS:
        cells.extend(
            case(
                f"{prefix}-SIZE-{size.upper()}",
                f"{size} canvas preset for this notebook's representative figure",
                source_template.replace("__SIZE__", size),
            )
        )


def notebook(title: str, introduction: str, cells: list, path: str) -> None:
    """Write one deterministic notebook."""
    nb = nbf.v4.new_notebook(
        cells=[markdown(f"# {title}\n\n{introduction}"), *cells],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "mlektic": {"generated": True, "generator": "scripts/generate_notebooks.py"},
        },
    )
    destination = NOTEBOOKS / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, destination)


def build_smoke() -> None:
    imports = """\
from IPython.display import display
from mlektic import visualize_lr, visualize_logistic
from notebooks._support import case_heading, linear_case, binary_case, multiclass_case
"""
    cells = setup(imports)
    cases = [
        (
            "SMOKE-LR-1D",
            "one-feature linear geometry and evolving equation",
            "c=linear_case(1)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=8,max_frames=6))",
        ),
        (
            "SMOKE-LR-2D",
            "two-feature regression plane",
            "c=linear_case(2)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=8,max_frames=6))",
        ),
        (
            "SMOKE-LR-ND",
            "six-feature symbolic summary without false geometry",
            "c=linear_case(6)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=8,max_frames=6,detail='academic'))",
        ),
        (
            "SMOKE-LOG-BIN-1D",
            "binary sigmoid, indexed classes, and optional loss",
            "c=binary_case(1)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=6,show_loss=True))",
        ),
        (
            "SMOKE-LOG-BIN-2D",
            "binary probability surface and decision boundary",
            "c=binary_case(2)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=6,show_loss=True))",
        ),
        (
            "SMOKE-LOG-BIN-ND",
            "high-dimensional binary symbolic route",
            "c=binary_case(6)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=6,show_loss=True))",
        ),
        (
            "SMOKE-LOG-MULTI-1D",
            "three indexed class-probability curves",
            "c=multiclass_case(1)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=6,show_loss=True))",
        ),
        (
            "SMOKE-LOG-MULTI-2D",
            "three class-probability surfaces",
            "c=multiclass_case(2)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=6,show_loss=True))",
        ),
        (
            "SMOKE-LOG-MULTI-ND",
            "six-feature multiclass matrix layout",
            "c=multiclass_case(6)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=6,show_loss=True))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "SMOKE",
        "c=linear_case(1)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=6,max_frames=4,size='__SIZE__'))",
    )
    notebook(
        "QA 00 — public route smoke matrix",
        "A fast visual pass across every tabular dimensional route.",
        cells,
        "qa/qa_00_smoke_matrix.ipynb",
    )


def build_linear() -> None:
    imports = """\
from IPython.display import display
from mlektic import explain_lr_prediction, visualize_lr
from notebooks._support import case_heading, linear_case, polynomial_linear_case, scaled_linear_case
"""
    cells = setup(imports)
    cases = [
        (
            "LR-CLOSED-INTERP",
            "closed-form estimator labeled as synthetic interpolation",
            "c=linear_case(1)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=18,max_frames=9,detail='academic'))",
        ),
        (
            "LR-SGD-REPLAY",
            "incremental replay ending at the exact fitted estimator",
            "c=linear_case(1,estimator='sgd')\ndisplay(visualize_lr(c.model,c.X,c.y,steps=24,max_frames=10,smooth='ema',detail='complete'))",
        ),
        (
            "LR-BASELINE-ZEROS",
            "zero parameter baseline instead of target-mean intercept",
            "c=linear_case(2)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=12,max_frames=6,baseline='zeros'))",
        ),
        (
            "LR-LOSS-OFF",
            "geometry fills the composition when loss is hidden",
            "c=linear_case(2)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=10,max_frames=6,show_loss=False))",
        ),
        (
            "LR-PIPE-ORIGINAL",
            "affine pipeline coefficients expressed in original units",
            "c=scaled_linear_case(2)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=10,max_frames=6,display_space='original',detail='complete'))",
        ),
        (
            "LR-PIPE-SCALED",
            "the same affine pipeline in scaled feature space",
            "c=scaled_linear_case(2)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=10,max_frames=6,display_space='scaled',detail='complete'))",
        ),
        (
            "LR-POLYNOMIAL",
            "curved raw-space geometry with transformed-space linear mathematics",
            "c=polynomial_linear_case()\ndisplay(visualize_lr(c.model,c.X,c.y,steps=12,max_frames=7,detail='complete',feature_names=['input']))",
        ),
        (
            "LR-MANY-FEATURES",
            "wrapped contribution mathematics for ten named features",
            "c=linear_case(10)\nnames=[f'feature_{i+1}_with_long_name' for i in range(10)]\ndisplay(visualize_lr(c.model,c.X,c.y,steps=8,max_frames=5,detail='complete',feature_names=names,sample_index=3,size='wide'))",
        ),
        (
            "LR-PRED-IN-RANGE",
            "in-range prediction point and high-contrast result label",
            "c=linear_case(1)\ndisplay(explain_lr_prediction(c.model,c.X,c.y,x_query=[[0.65]],theme='academic'))",
        ),
        (
            "LR-PRED-EXTRAP",
            "extrapolation notice and visible query outside the training range",
            "c=linear_case(2)\nquery=(c.X.max(axis=0)+1.5).reshape(1,-1)\ndisplay(explain_lr_prediction(c.model,c.X,c.y,x_query=query,size='wide'))",
        ),
        (
            "LR-PRED-10D-STAGES",
            "ten-variable prediction with distinct Input, Substitution, Output, and Reset controls",
            "c=linear_case(10)\ndisplay(explain_lr_prediction(c.model,c.X,c.y,x_query=c.X[[4]],size='wide'))",
        ),
        (
            "LR-PRED-COUNTERFACTUAL",
            "explicit user-provided counterfactual separated from model output",
            "c=linear_case(1)\ndisplay(explain_lr_prediction(c.model,c.X,c.y,x_query=[[0.2]],yhat=8.0,prediction_source='provided'))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "LR",
        "c=linear_case(1)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,detail='academic',size='__SIZE__'))",
    )
    notebook(
        "QA 01 — linear regression",
        "Linear estimators, feature spaces, mathematical density, and prediction edge cases.",
        cells,
        "qa/qa_01_linear.ipynb",
    )


def build_polynomial() -> None:
    """Build focused visual QA for polynomial geometry and equations."""
    imports = """\
import numpy as np
from IPython.display import display
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mlektic import visualize_lr
from notebooks._support import case_heading

SEED = 42
rng = np.random.default_rng(SEED)
"""
    cells = setup(imports)
    cells.extend(
        [
            markdown(
                """
## Qué valida este notebook

Una regresión polinómica conserva una entrada original 1D, pero transforma esa
entrada en términos como `x`, `x²`, `x³`, etc. Por eso la geometría visible debe
ser una **curva**, no una recta, y la ecuación superior debe mostrar las
potencias que realmente consume el estimador.

Revise en cada caso:

1. que el eje X use el `feature_names` solicitado;
2. que la línea del modelo tenga curvatura visible;
3. que la fórmula simbólica y la sustitución animada usen `x₁`, `x₁²`, `x₁³`, etc.;
4. que el último estado coincida con el modelo ajustado;
5. que grados altos se abrevien con `…` sin desbordar el lienzo.

Con `Ridge`/`LinearRegression`, los estados son una interpolación sintética
desde la línea base hasta la curva final. Con `SGDRegressor`, Mlektic reconstruye
un replay incremental y reserva el último estado para el estimador suministrado.
"""
            ),
            code(
                """X_quadratic = np.linspace(-2.8, 2.8, 110).reshape(-1, 1)
y_quadratic = 3.0 - 1.2 * X_quadratic[:, 0] + 2.4 * X_quadratic[:, 0] ** 2

X_waves = np.linspace(-3.0, 3.0, 180).reshape(-1, 1)
y_waves = (
    35 * np.sin(2.7 * X_waves[:, 0])
    + 16 * np.cos(5.2 * X_waves[:, 0])
    + 5 * X_waves[:, 0] ** 2
    + rng.normal(0, 4, len(X_waves))
)

diabetes = load_diabetes()
X_bmi = diabetes.data[:, [2]]
y_diabetes = diabetes.target

print('Parábola:', X_quadratic.shape)
print('Ondas:', X_waves.shape)
print('Diabetes/BMI:', X_bmi.shape)"""
            ),
            *case(
                "POLY-QUADRATIC-POWERS",
                "parábola inequívoca, eje descriptivo posición y ecuaciones canónicas x₁ y x₁²",
                """quadratic = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
).fit(X_quadratic, y_quadratic)
display(visualize_lr(
    quadratic, X_quadratic, y_quadratic,
    steps=16, max_frames=8, animation_mode='hybrid', interpolation_frames=4,
    detail='complete', feature_names=['posición'],
    title='QA polinómica: parábola y potencias explícitas', size='wide',
))""",
            ),
            *case(
                "POLY-CUBIC-SCALED-DIABETES",
                "curva cúbica real con eje BMI y ecuaciones canónicas x₁, x₁² y x₁³",
                """cubic_bmi = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    Ridge(alpha=4.0),
).fit(X_bmi, y_diabetes)
display(visualize_lr(
    cubic_bmi, X_bmi, y_diabetes,
    steps=20, max_frames=10, mode='final_interp',
    animation_mode='hybrid', interpolation_frames=4, fps=35,
    metrics=['mse', 'mae', 'r2'], detail='complete',
    feature_names=['BMI'], title='QA polinómica: eje BMI y potencias de x₁',
    theme='academic', size='wide',
))""",
            ),
            *case(
                "POLY-WAVY-HIGH-DEGREE",
                "deformación extrema desde una línea base hasta una curva con muchas oscilaciones",
                """wavy = make_pipeline(
    PolynomialFeatures(degree=14, include_bias=False),
    StandardScaler(),
    Ridge(alpha=0.08),
).fit(X_waves, y_waves)
display(visualize_lr(
    wavy, X_waves, y_waves,
    steps=40, max_frames=20, mode='final_interp',
    animation_mode='hybrid', interpolation_frames=5, fps=35,
    metrics=['mse', 'mae', 'r2'], detail='academic',
    feature_names=['posición'],
    title='QA polinómica: múltiples oscilaciones y grado 14 abreviado',
    theme='academic', size='wide',
))""",
            ),
            *case(
                "POLY-ITERATIVE-REPLAY",
                "aproximaciones reconstruidas de un polinomio incremental y cierre en el estimador exacto",
                """iterative_poly = make_pipeline(
    PolynomialFeatures(degree=5, include_bias=False),
    StandardScaler(),
    SGDRegressor(max_iter=1200, tol=1e-4, random_state=SEED),
).fit(X_waves, y_waves)
display(visualize_lr(
    iterative_poly, X_waves, y_waves,
    steps=24, max_frames=12, mode='iterative', animation_mode='native',
    smooth='ema', smooth_beta=0.9,
    metrics=['mse', 'mae', 'r2'], detail='complete',
    feature_names=['posición'],
    title='QA polinómica: replay incremental reconstruido',
    theme='accessible', size='wide',
))""",
            ),
            markdown(
                """
## Lectura esperada

- Los tres primeros casos no afirman mostrar pasos privados de entrenamiento:
  el subtítulo debe decir **Synthetic interpolation**.
- El caso incremental debe decir **Reconstructed replay** y terminar en
  **fitted**.
- `smooth='ema'` sólo afecta la pérdida del replay incremental. No se aplica a
  la interpolación sintética, que conserva el extremo exacto.
- En grado 14, la ecuación superior debe mostrar las primeras potencias,
  `\\cdots` y la potencia 14; la curva completa sí usa todos los coeficientes.
"""
            ),
        ]
    )
    add_size_coverage(
        cells,
        "POLY",
        "X_size=np.linspace(-2.8,2.8,110).reshape(-1,1)\ny_size=3.0-1.2*X_size[:,0]+2.4*X_size[:,0]**2\nm_size=make_pipeline(PolynomialFeatures(degree=2,include_bias=False),LinearRegression()).fit(X_size,y_size)\ndisplay(visualize_lr(m_size,X_size,y_size,steps=7,max_frames=5,feature_names=['posición'],size='__SIZE__'))",
    )
    notebook(
        "QA 09 — regresión polinómica y deformación de curvas",
        "Curvas 1D, potencias transformadas, interpolación sintética, replay incremental y límites de grado alto.",
        cells,
        "qa/qa_09_polynomial.ipynb",
    )


def build_logistic() -> None:
    imports = """\
from IPython.display import display
from mlektic import explain_logistic_prediction, visualize_logistic
from notebooks._support import case_heading, binary_case, multiclass_case, scaled_binary_case
"""
    cells = setup(imports)
    cases = [
        (
            "LOG-BINARY-INDEXED",
            "default class indices without semantic label noise",
            "c=binary_case(1,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=12,max_frames=7,show_loss=True,show_class_labels=False))",
        ),
        (
            "LOG-BINARY-LABELS",
            "opt-in semantic labels appended to class indices",
            "c=binary_case(1,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=12,max_frames=7,show_loss=True,show_class_labels=True))",
        ),
        (
            "LOG-BINARY-THRESHOLD",
            "non-default binary threshold with numeric tick and separate τ label inside the geometry",
            "c=binary_case(2)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,threshold=0.7,detail='academic'))",
        ),
        (
            "LOG-BINARY-IMBALANCED",
            "imbalanced data with empirical log-loss and metrics",
            "c=binary_case(2,imbalanced=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,show_loss=True,detail='complete'))",
        ),
        (
            "LOG-BINARY-30D-WRAP",
            "thirty-variable equation defines sigmoid once, wraps the intercept, and centers metric readouts",
            "c=binary_case(30)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=12,max_frames=6,show_loss=True,metrics=['log_loss','accuracy','f1'],detail='complete',format='compact',size='wide'))",
        ),
        (
            "LOG-SGD-REPLAY",
            "incremental classifier replay with EMA display smoothing",
            "c=binary_case(2,estimator='sgd')\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=20,max_frames=8,show_loss=True,smooth='ema'))",
        ),
        (
            "LOG-PIPE-ORIGINAL",
            "scaled pipeline explained in original units",
            "c=scaled_binary_case(2)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,display_space='original',detail='complete'))",
        ),
        (
            "LOG-PIPE-SCALED",
            "scaled pipeline coefficients in transformed units",
            "c=scaled_binary_case(2)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,display_space='scaled',detail='complete'))",
        ),
        (
            "LOG-MULTI-FOCUS",
            "one selected class surface while preserving fitted order",
            "c=multiclass_case(2,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,class_focus='group-1',show_class_labels=True,show_loss=True,detail='academic'))",
        ),
        (
            "LOG-MULTI-OVR",
            "normalized one-vs-rest semantics from SGDClassifier",
            "c=multiclass_case(2)\nfrom sklearn.linear_model import SGDClassifier\nm=SGDClassifier(loss='log_loss',max_iter=1000,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=12,max_frames=6,multiclass_link='ovr',show_loss=True,detail='academic'))",
        ),
        (
            "LOG-MULTI-MANY-CLASSES",
            "six-class matrix truncation and ellipsis",
            "c=multiclass_case(6,classes=6,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=5,max_theta_cols=3,detail='complete',size='wide'))",
        ),
        (
            "LOG-PRED-BINARY",
            "winning class index, both probabilities, curve, and boxed result",
            "c=binary_case(1,string_labels=True)\ndisplay(explain_logistic_prediction(c.model,c.X,c.y,x_query=[[0.4]],show_class_labels=False))",
        ),
        (
            "LOG-PRED-LABELS",
            "optional semantic winning label in prediction explanation",
            "c=binary_case(2,string_labels=True)\ndisplay(explain_logistic_prediction(c.model,c.X,c.y,x_query=c.X[[4]],show_class_labels=True,size='wide'))",
        ),
        (
            "LOG-PRED-MULTI",
            "multiclass probability vector and argmax winner",
            "c=multiclass_case(2,string_labels=True)\ndisplay(explain_logistic_prediction(c.model,c.X,c.y,x_query=c.X[[8]],show_class_labels=True))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    cells.append(markdown(
        """
## Reading the multiclass layouts

The animated coefficient matrices and equations display at most four decimal
places (`min(dec, 4)`); the fitted-model derivation below the figure honors
the requested `dec`. The short `log-loss` card uses the history source stated
in the subtitle and timeline; full loss provenance also remains in metadata.

`default`, `wide`, and `classroom` share the expanded multivariable notation:
endpoint scores on one line and numeric exponents in the probability fractions.
`compact` and `notebook` keep shorter symbolic fractions. Check the endpoint
continuation marks, the gap below the generic probability definition, and the
centering of the vertical ellipsis for both zero and fitted coefficients.
Omitted matrix columns are indicated by conventional ellipses, without an
extra caption. In binary 1D predictions, inspect the Substitution and Output
stages: the single-input panel gives the sigmoid derivation more vertical room.
"""
    ))
    add_size_coverage(
        cells,
        "LOG-ND",
        "c=multiclass_case(6,classes=6)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=5,show_loss=True,metrics=['log_loss','accuracy','f1'],max_theta_cols=2,detail='complete',size='__SIZE__'))",
    )
    notebook(
        "QA 02 — logistic regression",
        "Binary and multiclass links, labels, thresholds, imbalance, preprocessing, and predictions.",
        cells,
        "qa/qa_02_logistic.ipynb",
    )


def build_neural() -> None:
    imports = """\
from IPython.display import display
from mlektic import (explain_nn_prediction, visualize_nn, visualize_nn_architecture,
                     visualize_nn_graph, visualize_nn_training, visualize_nn_weights)
from notebooks._support import case_heading, torch_xor_case
model, X, history = torch_xor_case()
"""
    cells = setup(imports)
    cases = [
        (
            "NN-ARCHITECTURE",
            "layer roles, tensor dimensions, formulas, and optimizer metadata",
            "display(visualize_nn_architecture(model,X[:1],history=history,theme='academic'))",
        ),
        (
            "NN-GRAPH-EXACT",
            "exact activation colors, weight colors, and backpropagation overlays",
            "display(visualize_nn_graph(model,X[1],history,max_frames=6,frame_duration=220))",
        ),
        (
            "NN-GRAPH-RELATIVE",
            "relative activation contrast and forward-signal edge colors",
            "display(visualize_nn_graph(model,X[1],history,max_frames=6,node_color_mode='relative',edge_color_mode='signal',theme='accessible'))",
        ),
        (
            "NN-TRAINING",
            "recorded loss plus inferred accuracy, precision, and recall",
            "display(visualize_nn_training(history,max_frames=6))",
        ),
        (
            "NN-WEIGHTS",
            "truncated evolving matrices without truncating recorder values",
            "display(visualize_nn_weights(history,parameter='0.weight',max_rows=3,max_cols=2,max_frames=6))",
        ),
        (
            "NN-ACTIVATIONS",
            "recorded layer activation vectors over training",
            "display(visualize_nn(model,X[:1],history=history,view='activations',max_frames=6))",
        ),
        (
            "NN-PREDICTION",
            "fitted-model input, numerical substitution, and output stages",
            "display(explain_nn_prediction(model,X[1],history=history,parameter_state='final',theme='academic'))",
        ),
        (
            "NN-RELU-ADAM",
            "architecture and metadata for a different activation and optimizer",
            "m2,X2,h2=torch_xor_case(activation='relu',optimizer_name='adam',steps=8)\ndisplay(visualize_nn_architecture(m2,X2[:1],history=h2,format='compact',theme='classroom'))",
        ),
        (
            "NN-REPORT",
            "static reduced-motion final state for publication",
            "display(explain_nn_prediction(model,X[2],history=history,format='report',reduced_motion=True,size='wide'))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "NN",
        "display(visualize_nn_architecture(model,X[:1],history=history,theme='academic',size='__SIZE__'))",
    )
    notebook(
        "QA 03 — neural networks",
        "PyTorch architecture, genuine recorded training, parameters, activations, graph semantics, and predictions.",
        cells,
        "qa/qa_03_neural.ipynb",
    )


def build_neural_structures() -> None:
    imports = r"""
from IPython.display import display
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from mlektic import (TorchTrainingRecorder, explain_nn_prediction,
                     register_neural_descriptor, visualize_nn,
                     visualize_nn_architecture, visualize_nn_blocks,
                     visualize_nn_backpropagation, visualize_nn_graph,
                     visualize_nn_hyperparameters,
                     visualize_nn_loss_landscape, visualize_nn_training,
                     visualize_nn_weights)
from notebooks._support import case_heading, torch_xor_case

torch.manual_seed(17)

class ResidualNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.activation = torch.nn.ReLU()
        self.second = torch.nn.Linear(4, 4)
    def forward(self, x):
        return x + self.second(self.activation(self.first(x)))

class SharedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(4, 4)
    def forward(self, x):
        return self.shared(x) + self.shared(x)

class ConvNet(torch.nn.Module):
    def __init__(self, classes=3, in_channels=3):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Linear(4 * 4 * 4, classes)
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 6)
        self.output = torch.nn.Linear(6, 3)
    def forward(self, token_ids):
        return self.output(self.embedding(token_ids).mean(dim=1))

class SiameseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 3)
        self.head = torch.nn.Linear(6, 2)
    def forward(self, left, right):
        left_state = torch.relu(self.encoder(left))
        right_state = torch.relu(self.encoder(right))
        score = self.head(torch.cat((left_state, right_state), dim=-1))
        return score, left_state, right_state

class DynamicBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.positive = torch.nn.Linear(4, 2)
        self.negative = torch.nn.Linear(4, 2)
    def forward(self, x):
        return self.positive(x) if x.sum().item() > 0 else self.negative(x)

def record_case(model, X, y, loss_fn, *, task, steps=10, learning_rate=0.03):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recorder = TorchTrainingRecorder(
        model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        capture_optimizer_state=True,
    )
    for step in range(steps):
        optimizer.zero_grad()
        prediction = model(X)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        recorder.record(
            step + 1,
            loss=loss,
            predictions=prediction,
            targets=y,
            task=task,
            capture_phase="post_step",
        )
    recorder.close()
    return model, X, recorder.to_history()

# Small synthetic binary classification.
xor_model, xor_X, xor_history = torch_xor_case(steps=10)

# Synthetic nonlinear regression.
generator = torch.Generator().manual_seed(17)
reg_X = torch.rand((96, 2), generator=generator) * 4.0 - 2.0
reg_y = (0.7 * reg_X[:, :1] ** 2 - 0.4 * reg_X[:, 1:] + torch.sin(reg_X[:, :1]))
reg_model, reg_X, reg_history = record_case(
    torch.nn.Sequential(torch.nn.Linear(2, 10), torch.nn.Tanh(), torch.nn.Linear(10, 1)),
    reg_X,
    reg_y,
    torch.nn.MSELoss(),
    task="regression",
    steps=12,
)

# Real Iris multiclass classification (bundled with Scikit-learn; no download).
iris = load_iris()
iris_X = torch.tensor(StandardScaler().fit_transform(iris.data), dtype=torch.float32)
iris_y = torch.tensor(iris.target, dtype=torch.long)
iris_model, iris_X, iris_history = record_case(
    torch.nn.Sequential(torch.nn.Linear(4, 12), torch.nn.ReLU(), torch.nn.Linear(12, 3)),
    iris_X,
    iris_y,
    torch.nn.CrossEntropyLoss(),
    task="classification",
    steps=12,
)

# Real breast-cancer binary classification.
cancer = load_breast_cancer()
cancer_X = torch.tensor(StandardScaler().fit_transform(cancer.data), dtype=torch.float32)
cancer_y = torch.tensor(cancer.target[:, None], dtype=torch.float32)
cancer_model, cancer_X, cancer_history = record_case(
    torch.nn.Sequential(
        torch.nn.Linear(cancer_X.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(16, 1),
        torch.nn.Sigmoid(),
    ),
    cancer_X,
    cancer_y,
    torch.nn.BCELoss(),
    task="classification",
    steps=10,
)

# Real handwritten digits represented as 8x8 grayscale images.
digits = load_digits()
digits_X = torch.tensor(digits.images[:240, None] / 16.0, dtype=torch.float32)
digits_y = torch.tensor(digits.target[:240], dtype=torch.long)
digits_model, digits_X, digits_history = record_case(
    ConvNet(classes=10, in_channels=1),
    digits_X,
    digits_y,
    torch.nn.CrossEntropyLoss(),
    task="classification",
    steps=8,
    learning_rate=0.02,
)
"""
    cells = setup(imports)
    cells.append(
        markdown(
            """
This is the exhaustive neural figure gallery. It invokes every public Plotly
figure route with progressively richer models and both synthetic and real
datasets. The real datasets are bundled with Scikit-learn and require no
network download. Hover blocks to inspect shapes, parameters, buffers,
hyperparameters, readable mathematics, and capture provenance.

Training fixtures use short deterministic full-batch runs to create genuine
recorded states for visual inspection. They are not train/test benchmarks and
must not be interpreted as generalization estimates.

`build_nn_math_report`, `display_nn_math_report`, and `export_nn_math_report`
produce HTML reports rather than Plotly figures, so they remain covered by the
report tests and the dedicated neural documentation instead of this figure
gallery.
"""
        )
    )
    cells.append(markdown("## Every `visualize_nn` figure route — synthetic XOR"))
    routed_cases = [
        (
            "NN-ROUTER-ARCHITECTURE",
            "the generic router's legacy architecture view",
            "display(visualize_nn(xor_model,xor_X[:1],history=xor_history,view='architecture',theme='academic'))",
        ),
        (
            "NN-ROUTER-BLOCKS",
            "the generic router's execution-block view",
            "display(visualize_nn(xor_model,xor_X[:1],view='blocks',theme='classroom',size='wide'))",
        ),
        (
            "NN-ROUTER-HYPERPARAMETERS",
            "every effective model, optimizer-group, objective, and scheduler hyperparameter with its PyTorch-aligned mathematical definition",
            "hyper_model=torch.nn.Sequential(torch.nn.Linear(4,8,bias=False),torch.nn.BatchNorm1d(8,eps=1e-4,momentum=0.2),torch.nn.LeakyReLU(0.15),torch.nn.Dropout(0.25),torch.nn.Linear(8,3))\nhyper_optimizer=torch.optim.Adam(hyper_model.parameters(),lr=0.002,betas=(0.8,0.95),weight_decay=0.01)\nhyper_objective=torch.nn.CrossEntropyLoss(label_smoothing=0.05)\nhyper_scheduler=torch.optim.lr_scheduler.StepLR(hyper_optimizer,step_size=3,gamma=0.4)\ndisplay(visualize_nn_hyperparameters(hyper_model,optimizer=hyper_optimizer,loss_fn=hyper_objective,scheduler=hyper_scheduler,theme='academic',size='wide'))",
        ),
        (
            "NN-ROUTER-GRAPH",
            "the generic router's fluid hybrid animation with loss but without the optional backpropagation overlay",
            "display(visualize_nn(xor_model,xor_X[1],history=xor_history,view='graph',max_frames=None,frame_duration=360,evolution_mode='hybrid',update_reference='previous',update_scale='global',show_update_panel=False,show_loss_panel=True,show_backpropagation=False,top_k_updates=6,interpolation_frames=3,math_font_scale=1.15))",
        ),
        (
            "NN-ROUTER-TRAINING",
            "the generic router's recorded loss and metric panels",
            "display(visualize_nn(xor_model,history=xor_history,view='training',max_frames=6,theme='academic'))",
        ),
        (
            "NN-ROUTER-WEIGHTS",
            "the generic router's evolving parameter matrices",
            "display(visualize_nn(xor_model,history=xor_history,view='weights',max_frames=6,theme='academic'))",
        ),
        (
            "NN-ROUTER-ACTIVATIONS",
            "the generic router's recorded activation vectors",
            "display(visualize_nn(xor_model,xor_X[:1],history=xor_history,view='activations',max_frames=6,theme='accessible'))",
        ),
        (
            "NN-ROUTER-BACKPROPAGATION",
            "the generic router's chain-rule view with recorded layer-gradient norms",
            "display(visualize_nn(xor_model,xor_X[:1],history=xor_history,view='backpropagation',max_frames=6,theme='academic',math_font_scale=1.2))",
        ),
        (
            "NN-TRAINING-QUERY-REPLAY",
            "an independent replay of recorded parameter and signal evolution without prediction cards",
            "display(explain_nn_prediction(xor_model,xor_X[1],history=xor_history,max_frames=6,parameter_state='training_replay',theme='academic'))",
        ),
        (
            "NN-GALLERY-FORWARD-SUBSTITUTION",
            "a final-model input, numerical substitution, and output lesson for one XOR observation",
            "display(explain_nn_prediction(xor_model,xor_X[1],history=xor_history,parameter_state='final',theme='academic',size='wide'))",
        ),
        (
            "NN-GALLERY-GRAPH-SIGNAL",
            "smooth relative-activation contrast with forward-signal edge coloring",
            "display(visualize_nn_graph(xor_model,xor_X[1],xor_history,max_frames=None,frame_duration=360,interpolation_frames=3,node_color_mode='relative',edge_color_mode='signal',theme='accessible'))",
        ),
        (
            "NN-GALLERY-REPORT-FIGURE",
            "a static reduced-motion prediction figure for publication",
            "display(explain_nn_prediction(xor_model,xor_X[2],history=xor_history,format='report',reduced_motion=True,size='wide'))",
        ),
    ]
    for item in routed_cases:
        cells.extend(case(*item))

    cells.append(
        markdown("""
## Dense graph performance: without and with recorded backpropagation

The default graph omits the per-edge gradient overlay. This reduces animated
trace count while preserving forward activity, parameters, parameter updates,
and optional loss. The second case explicitly enables the dotted reverse-mode
gradient traces. Compare playback in the same notebook environment.
""")
    )
    cells.extend(
        case(
            "NN-GRAPH-WITHOUT-BACKPROP",
            "the default fluid graph with evolving parameters and loss but without per-edge gradient traces",
            "display(visualize_nn_graph(xor_model,xor_X[1],xor_history,max_frames=None,frame_duration=360,interpolation_frames=3,show_backpropagation=False,show_loss_panel=True,theme='academic',size='wide'))",
        )
    )
    cells.extend(
        case(
            "NN-GRAPH-WITH-BACKPROP",
            "the same graph with optional recorded reverse-mode gradients",
            "display(visualize_nn_graph(xor_model,xor_X[1],xor_history,max_frames=None,frame_duration=520,interpolation_frames=3,show_backpropagation=True,theme='academic',size='wide'))",
        )
    )

    cells.append(
        markdown("""
## Parameter-update evolution modes

The classic `absolute` graph remains the default. The following opt-in views
make small parameter changes perceptually visible without replacing their
mathematical values. A signed halo encodes the actual difference between the
current parameters and the selected reference; its width and opacity encode
the magnitude. Dashed edges continue to represent recorded gradients.

`update_scale="global"` keeps magnitudes comparable throughout the animation.
`update_scale="frame"` deliberately renormalizes every frame for contrast, so
its color intensity must not be compared across time. Interpolated frames make
motion smoother but are labeled as perceptual states, not optimizer steps; the
slider contains only recorded checkpoints.
""")
    )
    update_cases = [
        (
            "NN-GRAPH-HYBRID-UPDATES",
            "absolute weights plus globally comparable signed update halos and smooth perceptual motion",
            "display(visualize_nn_graph(xor_model,xor_X[1],xor_history,max_frames=None,frame_duration=360,evolution_mode='hybrid',update_reference='previous',update_scale='global',top_k_updates=6,interpolation_frames=3,theme='academic',size='wide'))",
        ),
        (
            "NN-GRAPH-CUMULATIVE-UPDATES",
            "parameter displacement from the initial checkpoint without the absolute-weight color encoding",
            "display(visualize_nn_graph(xor_model,xor_X[2],xor_history,max_frames=6,frame_duration=320,evolution_mode='updates',update_reference='initial',update_scale='global',interpolation_frames=2,theme='classroom',size='wide'))",
        ),
        (
            "NN-GRAPH-FRAME-NORMALIZED-UPDATES",
            "maximum per-frame contrast with an explicit warning that intensity is not comparable across time",
            "display(visualize_nn_graph(xor_model,xor_X[3],xor_history,max_frames=6,frame_duration=320,evolution_mode='updates',update_reference='previous',update_scale='frame',top_k_updates=4,theme='accessible',size='wide'))",
        ),
    ]
    for item in update_cases:
        cells.extend(case(*item))

    cells.append(
        markdown("""
## Objective geometry and backpropagation

The surface below is an exact evaluation of the selected loss on one affine
two-direction slice through parameter space. It is not the full
high-dimensional landscape. The recorded optimization path is projected onto
that plane. The backpropagation figure separately shows the canonical chain
rule and scales the backward paths by genuine recorded layer-gradient norms.
""")
    )
    cells.extend(
        case(
            "NN-LOSS-LANDSCAPE-XOR",
            "an exact BCELoss slice with the recorded XOR path projected onto it",
            "display(visualize_nn_loss_landscape(xor_model,xor_X,torch.tensor([[0.],[1.],[1.],[0.]]),torch.nn.BCELoss(),xor_history,grid_size=17,max_frames=6,theme='academic',size='wide'))",
        )
    )
    cells.extend(
        case(
            "NN-BACKPROP-XOR",
            "forward equations, backward chain rule, and globally comparable recorded gradient norms",
            "display(visualize_nn_backpropagation(xor_model,xor_history,input_sample=xor_X[:1],max_frames=None,frame_duration=1100,theme='classroom',math_font_scale=1.2,size='wide'))",
        )
    )

    cells.append(markdown("## Synthetic nonlinear regression"))
    regression_cases = [
        (
            "NN-SYNTH-REG-ARCHITECTURE",
            "a small dense regression model in the established architecture style",
            "display(visualize_nn_architecture(reg_model,reg_X[:1],history=reg_history,theme='academic'))",
        ),
        (
            "NN-SYNTH-REG-BLOCKS",
            "the same regression model as an execution graph with formulas",
            "display(visualize_nn_blocks(reg_model,reg_X[:1],theme='classroom',size='wide'))",
        ),
        (
            "NN-SYNTH-REG-TRAINING",
            "recorded MSE optimization and regression metrics",
            "display(visualize_nn_training(reg_history,max_frames=8,theme='academic'))",
        ),
        (
            "NN-SYNTH-REG-GRAPH",
            "animated hidden activations, weights, and gradients for regression",
            "display(visualize_nn_graph(reg_model,reg_X[3],reg_history,max_frames=8,frame_duration=260))",
        ),
        (
            "NN-SYNTH-REG-PREDICTION",
            "a final-model numerical regression substitution and prediction",
            "display(explain_nn_prediction(reg_model,reg_X[3],history=reg_history,parameter_state='final',theme='academic'))",
        ),
    ]
    for item in regression_cases:
        cells.extend(case(*item))

    cells.append(markdown("## Real Iris multiclass classification"))
    iris_cases = [
        (
            "NN-REAL-IRIS-ARCHITECTURE",
            "legacy architecture with real four-feature multiclass data",
            "display(visualize_nn_architecture(iris_model,iris_X[:1],history=iris_history,theme='classroom'))",
        ),
        (
            "NN-REAL-IRIS-BLOCKS",
            "execution blocks and a three-logit output",
            "display(visualize_nn_blocks(iris_model,iris_X[:1],theme='academic',size='wide'))",
        ),
        (
            "NN-REAL-IRIS-TRAINING",
            "cross-entropy with inferred multiclass metrics",
            "display(visualize_nn_training(iris_history,max_frames=8,theme='academic'))",
        ),
        (
            "NN-REAL-IRIS-WEIGHTS",
            "real-data parameter evolution with bounded matrix display",
            "display(visualize_nn_weights(iris_history,max_frames=8,max_rows=4,max_cols=5,theme='academic'))",
        ),
        (
            "NN-REAL-IRIS-ACTIVATIONS",
            "recorded hidden representations for multiclass learning",
            "display(visualize_nn(iris_model,iris_X[:1],history=iris_history,view='activations',max_frames=8,theme='accessible'))",
        ),
        (
            "NN-REAL-IRIS-GRAPH",
            "animated neural graph for one real Iris observation",
            "display(visualize_nn_graph(iris_model,iris_X[12],iris_history,max_frames=8,theme='classroom'))",
        ),
        (
            "NN-REAL-IRIS-PREDICTION",
            "layer-by-layer logits for one real Iris observation",
            "display(explain_nn_prediction(iris_model,iris_X[12],history=iris_history,max_frames=8,theme='academic'))",
        ),
    ]
    for item in iris_cases:
        cells.extend(case(*item))

    cells.append(markdown("## Real breast-cancer binary classification"))
    cancer_cases = [
        (
            "NN-REAL-CANCER-BLOCKS",
            "a wider 30-feature binary network with dropout and sigmoid",
            "display(visualize_nn_blocks(cancer_model,cancer_X[:1],theme='accessible',size='wide'))",
        ),
        (
            "NN-REAL-CANCER-TRAINING",
            "binary cross-entropy and inferred classification metrics",
            "display(visualize_nn_training(cancer_history,max_frames=8,theme='academic'))",
        ),
        (
            "NN-REAL-CANCER-PREDICTION",
            "a high-dimensional real-data binary forward explanation",
            "display(explain_nn_prediction(cancer_model,cancer_X[20],history=cancer_history,max_frames=7,max_neurons_math=6,theme='academic',size='wide'))",
        ),
    ]
    for item in cancer_cases:
        cells.extend(case(*item))

    cells.append(markdown("## Real handwritten digits with a convolutional network"))
    digit_cases = [
        (
            "NN-REAL-DIGITS-ARCHITECTURE",
            "legacy convolution, pooling, flatten, and ten-class architecture",
            "display(visualize_nn_architecture(digits_model,digits_X[:1],history=digits_history,max_layers=8,theme='classroom',size='wide'))",
        ),
        (
            "NN-REAL-DIGITS-BLOCKS",
            "shape-aware convolutional execution blocks for real 8x8 images",
            "display(visualize_nn_blocks(digits_model,digits_X[:1],theme='academic',size='wide'))",
        ),
        (
            "NN-REAL-DIGITS-DENSE-HEAD-GRAPH",
            "complete executed CNN topology; convolution, normalization, pooling, reshape, and classifier are all visible",
            "display(visualize_nn_graph(digits_model,digits_X[0],digits_history,max_neurons=8,max_frames=6,theme='academic',size='wide'))",
        ),
        (
            "NN-REAL-DIGITS-TRAINING",
            "recorded ten-class CNN training metrics",
            "display(visualize_nn_training(digits_history,max_frames=None,theme='academic'))",
        ),
        (
            "NN-REAL-DIGITS-WEIGHTS",
            "convolution kernels and dense matrices with explicit truncation",
            "display(visualize_nn_weights(digits_history,max_frames=None,max_rows=3,max_cols=4,max_parameters=5,theme='academic',size='wide'))",
        ),
    ]
    for item in digit_cases:
        cells.extend(case(*item))

    cells.append(markdown("## Structural and capture stress cases"))
    cases = [
        (
            "NN-BLOCK-RESIDUAL",
            "a residual branch and functional Add merge remain explicit",
            "m=ResidualNet()\nx=torch.randn(2,4)\ndisplay(visualize_nn_blocks(m,x,theme='academic',size='wide'))",
        ),
        (
            "NN-BLOCK-SHARED",
            "one shared Linear module appears as two ordered calls",
            "m=SharedNet()\nx=torch.randn(2,4)\ndisplay(visualize_nn_blocks(m,x,show_formulas=True,size='wide'))",
        ),
        (
            "NN-BLOCK-CONV",
            "convolution, BatchNorm buffers, activation, pooling, flatten, and classifier shapes",
            "m=ConvNet()\nx=torch.randn(2,3,8,8)\ndisplay(visualize_nn_blocks(m,x,theme='classroom',size='wide'))",
        ),
        (
            "NN-BLOCK-EMBEDDING",
            "integer token dtype, embedding, sequence reduction, and classifier",
            "m=EmbeddingNet()\ntokens=torch.tensor([[1,2,3,4],[3,5,7,9]],dtype=torch.long)\ndisplay(visualize_nn_blocks(m,tokens,theme='accessible',size='wide'))",
        ),
        (
            "NN-BLOCK-LSTM",
            "a recurrent primitive with hidden/cell outputs and public hyperparameters",
            "m=torch.nn.LSTM(5,7,num_layers=2,batch_first=True,bidirectional=True,dropout=0.1)\nx=torch.randn(2,4,5)\ndisplay(visualize_nn_blocks(m,x,theme='academic',size='wide'))",
        ),
        (
            "NN-BLOCK-ATTENTION",
            "query, key, value inputs and both attention outputs as one semantic primitive",
            "m=torch.nn.MultiheadAttention(8,2,batch_first=True,dropout=0.1)\nq=torch.randn(2,4,8)\ndisplay(visualize_nn_blocks(m,(q,q.clone(),q.clone()),theme='academic',size='wide'))",
        ),
        (
            "NN-BLOCK-MULTI-IO",
            "Siamese inputs, shared encoder calls, concatenation, score, and auxiliary outputs",
            "m=SiameseNet()\nleft,right=torch.randn(2,4),torch.randn(2,4)\ndisplay(visualize_nn_blocks(m,(left,right),theme='classroom',size='wide'))",
        ),
        (
            "NN-BLOCK-DYNAMIC",
            "data-dependent Python control flow uses the disclosed eager fallback",
            "m=DynamicBranch()\nx=torch.ones(2,4)\ndisplay(visualize_nn_blocks(m,x,theme='accessible',size='wide'))",
        ),
        (
            "NN-BLOCK-COLLAPSE",
            "a long network collapses visual middle nodes without changing capture",
            "layers=[]\nfor _ in range(18):\n    layers.extend([torch.nn.Linear(8,8),torch.nn.ReLU()])\nm=torch.nn.Sequential(*layers)\ndisplay(visualize_nn_blocks(m,torch.randn(2,8),max_nodes=12,show_formulas=False,size='wide'))",
        ),
        (
            "NN-BLOCK-CUSTOM",
            "a project-defined semantic descriptor extends the block vocabulary",
            "register_neural_descriptor('Identity',role='operation',label='Pedagogical identity',formula=r'\\mathbf{y}=\\mathbf{x}',replace=True)\nm=torch.nn.Identity()\ndisplay(visualize_nn_blocks(m,torch.randn(2,4),theme='academic'))",
        ),
        (
            "NN-RECORDER-V2",
            "buffers, optimizer groups, adaptive state norms, and temporal phases",
            "m=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4),torch.nn.ReLU(),torch.nn.Linear(4,1))\nx=torch.randn(12,4); y=torch.randn(12,1)\nopt=torch.optim.Adam([{'params':m[0].parameters(),'lr':0.01},{'params':list(m[1:].parameters()),'lr':0.003}],weight_decay=0.001)\nloss_fn=torch.nn.MSELoss()\nr=TorchTrainingRecorder(m,optimizer=opt,loss_fn=loss_fn,capture_optimizer_state=True)\nfor step in range(4):\n    opt.zero_grad(); prediction=m(x); loss=loss_fn(prediction,y); loss.backward(); opt.step(); r.record(step+1,loss=loss,predictions=prediction,targets=y,task='regression',capture_phase='post_step')\nr.close(); h=r.to_history()\ndisplay(visualize_nn_training(h,max_frames=None,theme='academic'))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))

    cells.append(
        markdown("""
## Hundreds of neurons and many layers

Large networks use semantic blocks rather than one glyph per neuron and one
line per connection. `max_nodes` bounds the semantic block view. A graph
request automatically prefers the complete executed topology whenever a dense
neuron replay would omit Dropout, convolution, normalization, pooling, tensor
operations, or branches. Pure dense graphs still sample up to `max_neurons`
values per layer and derive marker diameter from actual pixel spacing. This
keeps the visualization inspectable without presenting an incomplete network.
""")
    )
    large_setup = """widths=[128,512,384,256,128,64,32,10]
layers=[]
for left,right in zip(widths[:-1],widths[1:]):
    layers.extend([torch.nn.Linear(left,right),torch.nn.GELU(),torch.nn.Dropout(0.1)])
large_model=torch.nn.Sequential(*layers[:-1])
large_input=torch.randn(16,128)
large_target=torch.randint(0,10,(16,))
large_loss_fn=torch.nn.CrossEntropyLoss()
large_optimizer=torch.optim.Adam(large_model.parameters(),lr=0.002)
large_recorder=TorchTrainingRecorder(large_model,optimizer=large_optimizer,loss_fn=large_loss_fn,max_tensor_elements=300000,max_activation_elements=1024)
for step in range(6):
    large_optimizer.zero_grad(); large_prediction=large_model(large_input); large_loss=large_loss_fn(large_prediction,large_target); large_loss.backward(); large_optimizer.step()
    large_recorder.record(step+1,loss=large_loss,predictions=large_prediction,targets=large_target,task='classification')
large_recorder.close(); large_history=large_recorder.to_history()
display(visualize_nn_architecture(large_model,large_input[:1],history=large_history,max_layers=8,theme='academic',size='wide'))"""
    large_cases = [
        (
            "NN-LARGE-HUNDREDS-MANY-LAYERS",
            "a trained 128-to-512 network fixture reused by every supported neural figure",
            large_setup,
        ),
        (
            "NN-LARGE-ARCHITECTURE",
            "bounded mathematical architecture for hundreds of units",
            "display(visualize_nn_architecture(large_model,large_input[:1],history=large_history,max_layers=8,theme='academic',size='wide'))",
        ),
        (
            "NN-LARGE-BLOCKS",
            "execution graph with a concise collapsed summary node",
            "display(visualize_nn_blocks(large_model,large_input[:1],max_nodes=18,show_formulas=False,theme='academic',size='wide'))",
        ),
        (
            "NN-LARGE-GRAPH",
            "complete executed topology for the full deep network, including every Dropout stage",
            "display(visualize_nn_graph(large_model,large_input[0],large_history,max_neurons=8,max_frames=6,show_loss_panel=True,theme='academic',size='wide'))",
        ),
        (
            "NN-LARGE-FORWARD-SUBSTITUTION",
            "input, bounded layer substitutions, and ten-logit output",
            "display(explain_nn_prediction(large_model,large_input[0],history=large_history,max_layers_math=6,max_neurons_math=6,max_frames=6,theme='academic',size='wide'))",
        ),
        (
            "NN-LARGE-TRAINING",
            "cross-entropy and multiclass metrics",
            "display(visualize_nn_training(large_history,max_frames=None,theme='academic',size='wide'))",
        ),
        (
            "NN-LARGE-WEIGHTS",
            "bounded parameter matrices without reducing mathematical type",
            "display(visualize_nn_weights(large_history,max_rows=3,max_cols=4,max_parameters=6,max_frames=6,theme='academic',size='wide'))",
        ),
        (
            "NN-LARGE-ACTIVATIONS",
            "recorded layer activation summaries",
            "display(visualize_nn(large_model,large_input[:1],history=large_history,view='activations',max_frames=6,theme='accessible',size='wide'))",
        ),
        (
            "NN-LARGE-BACKPROPAGATION",
            "bounded per-layer gradients, updates, relative changes, and loss effect",
            "display(visualize_nn_backpropagation(large_model,large_history,input_sample=large_input[:1],max_layers=6,max_frames=6,frame_duration=1100,theme='classroom',size='wide'))",
        ),
        (
            "NN-LARGE-LOSS-LANDSCAPE",
            "exact reduced-grid two-direction CrossEntropyLoss slice for the large model",
            "display(visualize_nn_loss_landscape(large_model,large_input,large_target,large_loss_fn,large_history,grid_size=9,max_frames=4,theme='academic',size='wide'))",
        ),
    ]
    for item in large_cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "NN-GALLERY",
        "display(visualize_nn_architecture(xor_model,xor_X[:1],history=xor_history,theme='academic',size='__SIZE__'))",
    )
    notebook(
        "QA 08 — complete neural figure gallery",
        "Every public neural Plotly figure, synthetic and real datasets, simple and complex topology, extensibility, scale, fallbacks, and recorder schema v2.",
        cells,
        "qa/qa_08_neural_structures.ipynb",
    )


def build_motion() -> None:
    imports = """\
from IPython.display import display
from mlektic import visualize_lr, visualize_logistic
from notebooks._support import case_heading, binary_case, linear_case
linear = linear_case(1,estimator='sgd')
logistic = binary_case(1,estimator='sgd')
"""
    cells = setup(imports)
    cells.append(
        markdown("""
`steps=K` creates semantic checkpoints. `max_frames=N` retains at most N of
those checkpoints. `interpolation_frames=q` inserts perceptual intervals in
hybrid linear motion; they are not optimizer updates. `frame_duration` controls
playback time, never whether the model trace exists. The dashboard retains the
complete composition and its temporal controls.
""")
    )
    cases = [
        (
            "MOTION-NATIVE-SLOW",
            "slow native playback with 8 displayed checkpoints",
            "display(visualize_lr(linear.model,linear.X,linear.y,steps=24,max_frames=8,animation_mode='native',frame_duration=500))",
        ),
        (
            "MOTION-HYBRID-FLUID",
            "fluid hybrid subframes while labels advance only at checkpoints",
            "display(visualize_lr(linear.model,linear.X,linear.y,steps=24,max_frames=8,animation_mode='hybrid',interpolation_frames=4,fps=35))",
        ),
        (
            "MOTION-FRAME-STEP",
            "source stride when max_frames is disabled",
            "display(visualize_lr(linear.model,linear.X,linear.y,steps=25,max_frames=None,frame_step=4,animation_mode='native'))",
        ),
        (
            "MOTION-NO-SMOOTH",
            "raw replay loss without EMA",
            "display(visualize_lr(linear.model,linear.X,linear.y,steps=20,max_frames=8,smooth=None,animation_mode='native'))",
        ),
        (
            "MOTION-LOGISTIC",
            "logistic transition duration and redraw-safe model visibility",
            "display(visualize_logistic(logistic.model,logistic.X,logistic.y,steps=18,max_frames=7,show_loss=True,frame_duration=420,transition_duration=280))",
        ),
        (
            "MOTION-REDUCED",
            "exact final state without playback controls",
            "display(visualize_lr(linear.model,linear.X,linear.y,steps=18,max_frames=7,reduced_motion=True,detail='academic'))",
        ),
        (
            "MOTION-REPORT",
            "static report composition as an explicit non-animated format",
            "display(visualize_logistic(logistic.model,logistic.X,logistic.y,steps=18,max_frames=7,format='report',show_loss=True))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "MOTION",
        "display(visualize_lr(linear.model,linear.X,linear.y,steps=12,max_frames=6,animation_mode='native',size='__SIZE__'))",
    )
    notebook(
        "QA 04 — motion and temporal semantics",
        "Playback controls, decimation, smoothing, complete dashboard composition, and static accessibility modes.",
        cells,
        "qa/qa_04_motion.ipynb",
    )


def build_visual_system() -> None:
    imports = """\
from IPython.display import display
from mlektic import visualize_lr
from notebooks._support import case_heading, linear_case
c=linear_case(1)
"""
    cells = setup(imports)
    for theme in ["classic", "academic", "classroom", "compact", "accessible"]:
        cells.extend(
            case(
                f"STYLE-THEME-{theme.upper()}",
                f"{theme} visual tokens without changing model semantics",
                f"display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,theme='{theme}'))",
            )
        )
    for fmt in ["dashboard", "compact", "report"]:
        cells.extend(
            case(
                f"STYLE-FORMAT-{fmt.upper()}",
                f"{fmt} composition and its documented motion behavior",
                f"display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,format='{fmt}',theme='academic'))",
            )
        )
    for size in ["default", "compact", "notebook", "wide", "classroom"]:
        cells.extend(
            case(
                f"STYLE-SIZE-{size.upper()}",
                f"{size} canvas preset and readable labels",
                f"display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,size='{size}',theme='academic'))",
            )
        )
    cases = [
        (
            "STYLE-RESPONSIVE",
            "responsive config metadata and container scaling",
            "display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,responsive=True,size='notebook'))",
        ),
        (
            "STYLE-EXPLICIT-SIZE",
            "explicit width and height overriding the preset",
            "display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,width=960,height=640))",
        ),
        (
            "STYLE-DENSITY-ESSENTIAL",
            "compact essential mathematics",
            "display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,density='essential'))",
        ),
        (
            "STYLE-DENSITY-ACADEMIC",
            "fitted-model derivation below the animation",
            "display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,density='academic',size='wide'))",
        ),
        (
            "STYLE-DENSITY-COMPLETE",
            "objective, preprocessing, and caveat panels",
            "display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,density='complete',size='wide'))",
        ),
        (
            "STYLE-HISTORY-HIDDEN",
            "history subtitle hidden while slider and metadata retain provenance",
            "display(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,show_history_context=False))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    notebook(
        "QA 05 — themes, formats, density, and sizes",
        "Every supported visual-system preset plus overrides and responsive metadata.",
        cells,
        "qa/qa_05_visual_system.ipynb",
    )


def build_hyperparameters() -> None:
    imports = """\
import numpy as np
from IPython.display import display
from sklearn.linear_model import LogisticRegression, SGDRegressor
from mlektic import TorchTrainingRecorder, visualize_lr, visualize_logistic, visualize_nn_architecture, visualize_nn_training
from notebooks._support import binary_case, case_heading, linear_case, multiclass_case
"""
    cells = setup(imports)
    cases = [
        (
            "HYPER-LR-L2",
            "SGDRegressor L2 regularization strength exposed in the complete panel",
            "c=linear_case(2)\nm=SGDRegressor(penalty='l2',alpha=0.001,max_iter=500,random_state=17).fit(c.X,c.y)\ndisplay(visualize_lr(m,c.X,c.y,steps=16,max_frames=7,detail='complete'))",
        ),
        (
            "HYPER-LR-L1",
            "SGDRegressor L1 penalty and sparse-coefficient tendency",
            "c=linear_case(6)\nm=SGDRegressor(penalty='l1',alpha=0.02,max_iter=800,random_state=17).fit(c.X,c.y)\ndisplay(visualize_lr(m,c.X,c.y,steps=16,max_frames=7,detail='complete',size='wide'))",
        ),
        (
            "HYPER-LR-LEARNING-RATE",
            "constant learning-rate replay metadata without claiming original fit recovery",
            "c=linear_case(1)\nm=SGDRegressor(learning_rate='constant',eta0=0.01,max_iter=300,random_state=17).fit(c.X,c.y)\ndisplay(visualize_lr(m,c.X,c.y,steps=18,max_frames=8,detail='complete'))",
        ),
        (
            "HYPER-LOG-C-STRONG",
            "stronger logistic L2 regularization through a smaller C",
            "c=binary_case(2)\nm=LogisticRegression(C=0.1,max_iter=1000,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=12,max_frames=6,detail='complete',show_loss=True))",
        ),
        (
            "HYPER-LOG-C-WEAK",
            "weaker logistic L2 regularization through a larger C",
            "c=binary_case(2)\nm=LogisticRegression(C=100.0,max_iter=1000,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=12,max_frames=6,detail='complete',show_loss=True))",
        ),
        (
            "HYPER-LOG-L1",
            "binary L1 penalty with a compatible public solver",
            "c=binary_case(6)\nm=LogisticRegression(penalty='l1',solver='liblinear',C=0.5,max_iter=1000,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=12,max_frames=6,detail='complete',size='wide'))",
        ),
        (
            "HYPER-LOG-CLASS-WEIGHT",
            "balanced class weights on an imbalanced dataset",
            "c=binary_case(2,imbalanced=True)\nm=LogisticRegression(class_weight='balanced',max_iter=1000,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=12,max_frames=6,detail='complete',show_loss=True))",
        ),
        (
            "HYPER-LOG-MULTICLASS-C",
            "multiclass regularization and complete matrix mathematics",
            "c=multiclass_case(3,classes=4)\nm=LogisticRegression(C=0.4,max_iter=1200,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=10,max_frames=6,detail='complete',show_loss=True))",
        ),
        (
            "HYPER-NN-DEEP",
            "deeper dense architecture with BatchNorm, ReLU, and Dropout roles",
            "import torch\ntorch.manual_seed(17)\nm=torch.nn.Sequential(torch.nn.Linear(6,12),torch.nn.BatchNorm1d(12),torch.nn.ReLU(),torch.nn.Dropout(0.2),torch.nn.Linear(12,4),torch.nn.Tanh(),torch.nn.Linear(4,1))\nm.eval()\ndisplay(visualize_nn_architecture(m,torch.zeros(4,6),max_layers=10,theme='academic',size='wide'))",
        ),
        (
            "HYPER-NN-CONV",
            "convolution, activation, flattening, and dense output dimensions",
            "import torch\ntorch.manual_seed(17)\nm=torch.nn.Sequential(torch.nn.Conv2d(1,3,kernel_size=3),torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(108,2))\ndisplay(visualize_nn_architecture(m,torch.zeros(1,1,8,8),max_layers=8,theme='classroom',size='wide'))",
        ),
        (
            "HYPER-NN-REGRESSION",
            "recorded MSE training for a neural regression task",
            "import torch\ntorch.manual_seed(17)\nX=torch.linspace(-1,1,20).reshape(-1,1); y=1+2*X\nm=torch.nn.Sequential(torch.nn.Linear(1,6),torch.nn.Tanh(),torch.nn.Linear(6,1))\nopt=torch.optim.Adam(m.parameters(),lr=0.05); loss_fn=torch.nn.MSELoss(); rec=TorchTrainingRecorder(m,optimizer=opt,loss_fn=loss_fn)\nfor step in range(10):\n opt.zero_grad(); pred=m(X); loss=loss_fn(pred,y); loss.backward(); opt.step(); rec.record(step+1,loss=loss,predictions=pred,targets=y,task='regression')\nrec.close(); display(visualize_nn_training(rec.to_history(),max_frames=8,theme='academic'))",
        ),
        (
            "HYPER-NN-MULTICLASS",
            "recorded CrossEntropy training and inferred multiclass metrics",
            "import torch\ntorch.manual_seed(17)\nX=torch.randn(24,3); y=torch.arange(24)%3\nm=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),torch.nn.Linear(8,3))\nopt=torch.optim.SGD(m.parameters(),lr=0.1); loss_fn=torch.nn.CrossEntropyLoss(); rec=TorchTrainingRecorder(m,optimizer=opt,loss_fn=loss_fn)\nfor step in range(8):\n opt.zero_grad(); pred=m(X); loss=loss_fn(pred,y); loss.backward(); opt.step(); rec.record(step+1,loss=loss,predictions=pred,targets=y,task='classification')\nrec.close(); display(visualize_nn_training(rec.to_history(),max_frames=8,theme='classroom'))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "HYPER",
        "c=linear_case(2)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=7,max_frames=5,detail='complete',size='__SIZE__'))",
    )
    notebook(
        "QA 07 — model parameters and hyperparameters",
        "Estimator-owned regularization, optimization, weighting, topology, activation, loss, and task variants.",
        cells,
        "qa/qa_07_hyperparameters.ipynb",
    )


def build_data_edges() -> None:
    imports = """\
import numpy as np
from IPython.display import display
from sklearn.linear_model import LinearRegression, LogisticRegression
from mlektic import visualize_lr, visualize_logistic
from notebooks._support import case_heading, linear_case, multiclass_case
"""
    cells = setup(imports)
    cases = [
        (
            "DATA-LR-NOISELESS",
            "exact fit and near-zero endpoint metrics",
            "c=linear_case(2,noise=0.0)\ndisplay(visualize_lr(c.model,c.X,c.y,steps=8,max_frames=5,detail='complete'))",
        ),
        (
            "DATA-LR-SCALE",
            "features with radically different numeric scales",
            "rng=np.random.default_rng(17)\nX=np.column_stack([rng.normal(size=80),1000*rng.normal(size=80)])\ny=1+2*X[:,0]-0.004*X[:,1]\nm=LinearRegression().fit(X,y)\ndisplay(visualize_lr(m,X,y,steps=8,max_frames=5,detail='academic'))",
        ),
        (
            "DATA-LR-CONSTANT-TARGET",
            "constant target without divide-by-zero visual artifacts",
            "X=np.linspace(-2,2,60).reshape(-1,1)\ny=np.full(60,3.5)\nm=LinearRegression().fit(X,y)\ndisplay(visualize_lr(m,X,y,steps=8,max_frames=5,detail='complete'))",
        ),
        (
            "DATA-LOG-OVERLAP",
            "overlapping binary classes and calibrated probability curve",
            "rng=np.random.default_rng(19)\nX=rng.normal(size=(100,1))\ny=(X[:,0]+rng.normal(size=100)>0).astype(int)\nm=LogisticRegression(max_iter=1000).fit(X,y)\ndisplay(visualize_logistic(m,X,y,steps=8,max_frames=5,show_loss=True))",
        ),
        (
            "DATA-MULTI-FOUR",
            "four-class probability geometry and legend",
            "c=multiclass_case(2,classes=4,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=8,max_frames=5,class_focus='group-2',show_class_labels=True))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    add_size_coverage(
        cells,
        "DATA",
        "c=multiclass_case(2,classes=4,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=7,max_frames=5,class_focus='group-2',show_class_labels=True,size='__SIZE__'))",
    )
    notebook(
        "QA 06 — data regimes and edge cases",
        "Numerical scale, noise, constant targets, overlap, and additional class counts.",
        cells,
        "qa/qa_06_data_edges.ipynb",
    )


def lesson_common(title: str, objective: str, code_cells: list, filename: str) -> None:
    imports = """\
import numpy as np
from IPython.display import display
from notebooks._support import case_heading
"""
    cells = setup(imports)
    cells.extend(code_cells)
    notebook(title, objective, cells, f"learn/{filename}")


def build_learning() -> None:
    lesson_common(
        "Learn 00 — getting started",
        "Fit one model, identify where animation states come from, and inspect one prediction.",
        [
            markdown("## 1. Fit a model\n\nThe estimator is already fitted before Mlektic receives it."),
            code(
                "from sklearn.linear_model import LinearRegression\nfrom mlektic import explain_lr_prediction, visualize_lr\nX=np.linspace(-2,2,60).reshape(-1,1)\ny=1.2+2.0*X[:,0]\nmodel=LinearRegression().fit(X,y)"
            ),
            *case(
                "LEARN-GETTING-STARTED",
                "synthetic interpolation is labeled and ends at the fitted model",
                "display(visualize_lr(model,X,y,steps=12,max_frames=7,theme='classroom'))",
            ),
            markdown(
                "## 2. Explain a prediction\n\nCompare the symbolic rule, numerical substitution, and plotted output."
            ),
            *case(
                "LEARN-GETTING-PREDICTION",
                "one model-verified prediction",
                "display(explain_lr_prediction(model,X,y,x_query=[[0.75]],theme='classroom'))",
            ),
            markdown(
                "## Reflection\n\nWhich states are exact fitted states? Which are synthetic? What does changing `frame_duration` alter mathematically?"
            ),
        ],
        "learn_00_getting_started.ipynb",
    )
    lesson_common(
        "Learn 01 — linear regression mathematics",
        "Connect coefficients, geometry, empirical error metrics, and original/transformed feature spaces.",
        [
            code(
                "from mlektic import explain_lr_prediction, visualize_lr\nfrom notebooks._support import linear_case, polynomial_linear_case\nc=linear_case(2)"
            ),
            markdown("## Linear form\n\nFor two features, $\\hat y=\\theta_0+\\theta_1x_1+\\theta_2x_2$ is a plane."),
            *case(
                "LEARN-LINEAR-PLANE",
                "coefficients, plane geometry, loss, and fitted derivation agree",
                "display(visualize_lr(c.model,c.X,c.y,steps=14,max_frames=8,detail='complete'))",
            ),
            markdown("## Linear in parameters is not always a straight raw-space line"),
            *case(
                "LEARN-LINEAR-POLY",
                "polynomial features produce a curve but remain linear in theta",
                "p=polynomial_linear_case()\ndisplay(visualize_lr(p.model,p.X,p.y,steps=12,max_frames=7,detail='complete'))",
            ),
            markdown(
                "## Reflection\n\nExplain why interpolation MSE is not an optimizer loss and why the fitted endpoint is nevertheless exact."
            ),
        ],
        "learn_01_linear_regression.ipynb",
    )
    lesson_common(
        "Learn 02 — logistic regression mathematics",
        "Connect linear scores, sigmoid/Softmax probabilities, thresholds, and winning class indices.",
        [
            code(
                "from mlektic import explain_logistic_prediction, visualize_logistic\nfrom notebooks._support import binary_case, multiclass_case\nb=binary_case(2,string_labels=True)"
            ),
            markdown(
                "## Binary probability\n\nThe fitted class order defines $p_0$ and $p_1$; semantic names are optional presentation."
            ),
            *case(
                "LEARN-LOGISTIC-BINARY",
                "score, sigmoid probability, threshold, and boundary",
                "display(visualize_logistic(b.model,b.X,b.y,steps=14,max_frames=8,threshold=0.65,detail='complete',show_loss=True))",
            ),
            *case(
                "LEARN-LOGISTIC-PREDICTION",
                "winning class index and optional semantic label",
                "display(explain_logistic_prediction(b.model,b.X,b.y,x_query=b.X[[5]],show_class_labels=True))",
            ),
            markdown(
                r"""
## Multiclass probability and indices

Here $K$ counts fitted classes, $k=0,\ldots,K-1$ indexes a class in `model.classes_`,
and $d$ counts input features. For every class, including those abbreviated
by $\vdots$ in the figure, use the same two equations:

$$z_k(\mathbf{x})=\theta_{0,k}+\sum_{j=1}^{d}\theta_{j,k}x_j,\qquad
\hat p(Y=c_k\mid\mathbf{x})=\frac{e^{z_k}}{\sum_{\ell=0}^{K-1}e^{z_\ell}}.$$

For $d=1$, $z_k=\theta_{0,k}+\theta_{1,k}x_1$. For $d=2$, add
$\theta_{2,k}x_2$ to that *same* score formula. As a numerical example with
$K=3$, $x_1=2$, intercepts $(0.2,-0.4,0)$, and coefficient columns
$(1,-0.5,0.25)$, the scores are $(2.2,-1.4,0.5)$. Therefore
$\hat p(Y=c_0\mid\mathbf{x})=e^{2.2}/(e^{2.2}+e^{-1.4}+e^{0.5})\approx0.8264$;
repeat the substitution with each remaining column for $c_1$ and $c_2$.
With $x_2=1$ and second-feature coefficients $(0.1,0.4,-0.2)$, the scores
become $(2.3,-1.0,0.3)$ and the same Softmax rule gives
$(0.8531,0.0315,0.1155)$. These numbers illustrate the notation; the plot
uses fitted coefficients. Binary logistic regression instead displays one
logit and a sigmoid for its positive class.
"""
            ),
            *case(
                "LEARN-LOGISTIC-MULTI",
                "Softmax probabilities sum to one and argmax selects a class",
                "m=multiclass_case(2,string_labels=True)\ndisplay(visualize_logistic(m.model,m.X,m.y,steps=12,max_frames=7,class_focus='group-1',show_class_labels=True,detail='academic'))",
            ),
        ],
        "learn_02_logistic_regression.ipynb",
    )
    lesson_common(
        "Learn 03 — neural-network mathematics",
        "Read layer dimensions, forward substitutions, recorded metrics, parameters, and gradients.",
        [
            code(
                "from mlektic import explain_nn_prediction, visualize_nn_architecture, visualize_nn_graph, visualize_nn_training\nfrom notebooks._support import torch_xor_case\nmodel,X,history=torch_xor_case(steps=10)"
            ),
            markdown("## Architecture before animation"),
            *case(
                "LEARN-NN-ARCHITECTURE",
                "each layer's tensor dimensions and mathematical role",
                "display(visualize_nn_architecture(model,X[:1],history=history,theme='classroom'))",
            ),
            markdown(
                "## Recorded training\n\nThese checkpoints were captured by `TorchTrainingRecorder`; they are not reconstructed."
            ),
            *case(
                "LEARN-NN-TRAINING",
                "recorded loss and classification metrics",
                "display(visualize_nn_training(history,max_frames=8))",
            ),
            *case(
                "LEARN-NN-GRAPH",
                "forward activations and backpropagation quantities",
                "display(visualize_nn_graph(model,X[1],history,max_frames=8,frame_duration=260))",
            ),
            *case(
                "LEARN-NN-PREDICTION",
                "final fitted-model input, substitution, and output stages",
                "display(explain_nn_prediction(model,X[1],history=history,max_frames=8))",
            ),
        ],
        "learn_03_neural_networks.ipynb",
    )

    lesson_common(
        "Learn 04 — neural execution structures",
        "Connect residual paths, repeated modules, attention, and capture provenance to executable tensor mathematics.",
        [
            code(
                """from mlektic import inspect_nn, visualize_nn_blocks
import torch

class LessonResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(4, 4)
        self.output = torch.nn.Linear(4, 4)
    def forward(self, x):
        return x + self.output(torch.relu(self.hidden(x)))

torch.manual_seed(17)
x=torch.randn(2,4)"""
            ),
            markdown(
                "## Residual paths\n\nThe tensor graph separates the transformed path from the identity path and joins them at Add."
            ),
            *case(
                "LEARN-NN-BRANCHES",
                "read a residual branch, tensor dimensions, and the merge equation",
                "model=LessonResidual()\ndisplay(visualize_nn_blocks(model,x,theme='classroom',size='wide'))",
            ),
            markdown(
                "## Attention as a semantic primitive\n\nQuery, key, and value are distinct argument ports. Hover the attention block to find embed_dim, num_heads, dropout, and output shapes."
            ),
            *case(
                "LEARN-NN-ATTENTION",
                "connect Q, K, V inputs to scaled dot-product multi-head attention",
                "attention=torch.nn.MultiheadAttention(8,2,batch_first=True)\nq=torch.randn(2,4,8)\ndisplay(visualize_nn_blocks(attention,(q,q.clone(),q.clone()),theme='academic',size='wide'))",
            ),
            markdown(
                "## Capture provenance\n\nFX can retain supported functional operations. Eager hooks record executed module calls when static tracing is not possible. Neither route claims to prove every possible dynamic path."
            ),
            *case(
                "LEARN-NN-PROVENANCE",
                "compare the visible figure with the renderer-independent graph contract",
                "graph=inspect_nn(model,x)\ndisplay(visualize_nn_blocks(model,x,theme='accessible',show_formulas=False,size='wide'))",
            ),
        ],
        "learn_04_neural_architectures.ipynb",
    )


def build_realistic_scenarios() -> None:
    """Build the end-to-end realistic scenario and configuration reference."""
    imports = """\
from IPython.display import display
import inspect
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from mlektic import (
    TorchTrainingRecorder,
    explain_logistic_prediction,
    explain_lr_prediction,
    explain_nn_prediction,
    inspect_nn,
    visualize_logistic,
    visualize_lr,
    visualize_nn,
    visualize_nn_architecture,
    visualize_nn_backpropagation,
    visualize_nn_blocks,
    visualize_nn_graph,
    visualize_nn_hyperparameters,
    visualize_nn_loss_landscape,
    visualize_nn_training,
    visualize_nn_weights,
)
from notebooks._support import case_heading

SEED = 17
np.random.seed(SEED)
"""
    cells = setup(imports)
    cells.extend(
        [
            markdown(
                """
## Cómo usar este laboratorio

Cada sección combina un conjunto de datos incluido en scikit-learn (sin descargas), un modelo reproducible y las figuras públicas que corresponden a su dimensionalidad. Ejecute las celdas en orden. Los ejemplos reducen pasos y muestras para que la exploración sea ágil; aumente `steps`, `max_frames` o los pasos de entrenamiento cuando quiera mayor resolución.

**Lectura honesta del tiempo:** un estimador ya ajustado no conserva su trayectoria privada. `mode='auto'` reconstruye un replay cuando el estimador es incremental y, en los demás casos, muestra interpolación sintética hasta el punto final exacto. En redes, el historial sí es registrado durante el entrenamiento por `TorchTrainingRecorder`.
"""
            ),
            markdown(
                """
## Configuración visual compartida por todas las familias

Estos argumentos controlan presentación, no el modelo ni sus predicciones. La columna **Cuándo tocarlo** sirve como guía de exploración.

| Parámetro | Valores admitidos | Qué hace | Cuándo tocarlo |
|---|---|---|---|
| `theme` | `None`/`'classic'`, `'academic'`, `'classroom'`, `'compact'`, `'accessible'` | Cambia colores, tipografía, contrastes, grosores y redundancia visual; no elimina información matemática. `'classroom'` se conserva como alias compatible de `'academic'`: el tamaño de pantalla se controla con `size`, no agrandando ejes y ecuaciones. | Compare `'academic'` para lectura técnica, `'compact'` para poco espacio y `'accessible'` cuando el color no deba ser la única señal. |
| `format` | `'dashboard'`, `'compact'`, `'report'` | Controla la composición. `'dashboard'` mantiene la vista interactiva completa y los controles propios de cada figura; `'compact'` reduce espacio y `'report'` fija el estado final sin controles. | Use `'dashboard'` al explorar y `'report'` para exportar/capturar una figura estática. |
| `detail` | `'essential'`, `'academic'`, `'complete'` | En lineal/logística decide cuánta derivación, diagnóstico y contexto se muestran. | Suba a `'complete'` para auditar todos los paneles; vuelva a `'essential'` para comprobar la vista mínima. |
| `density` | `None` o `'essential'`, `'academic'`, `'complete'` | Alias de densidad visual; en lineal/logística complementa `detail` y en redes es el nombre principal. No debe contradecir un `detail` explícito no predeterminado. | Úselo para probar cuánto contenido cabe en cada tamaño de lienzo. |
| `size` | `'default'`, `'compact'`, `'notebook'`, `'wide'`, `'classroom'` | Selecciona un preset de lienzo y espaciado. `'classroom'` se conserva como alias compatible de `'wide'`. | Pruebe `'notebook'` en Jupyter y `'wide'` con matrices, fórmulas largas o una pantalla grande. |
| `width` | `None` o entero `>=320` | Sustituye sólo el ancho del preset de `size`. | Fuerce anchos estrechos y amplios para detectar solapamientos y saltos de texto. |
| `height` | `None` o entero `>=320` | Sustituye sólo la altura del preset de `size`. | Auméntelo si hay muchos paneles; redúzcalo para probar el límite vertical. |
| `responsive` | `False`/`True` | Hace que el HTML escale con el contenedor manteniendo su relación interna. | Actívelo en notebooks o páginas cuyo ancho cambia; no sustituye elegir una composición adecuada. |
| `reduced_motion` | `False`/`True` | Desactiva la reproducción, retira controles temporales y deja un estado final legible. | Valide accesibilidad o genere una vista estable sin modificar el historial. |
| `title` | `None` o texto | Sustituye el título automático de la figura; `None` conserva el nombre generado a partir del caso. | Compruebe títulos largos, caracteres Unicode y nombres de dominio reales. |

Use `format='report'` para exportación estática, `theme='accessible'` cuando el color no deba ser la única señal y `size='wide'` para fórmulas o matrices grandes.
"""
            ),
            code(
                """public_figures = [
    visualize_lr, explain_lr_prediction, visualize_logistic, explain_logistic_prediction,
    visualize_nn, visualize_nn_architecture, visualize_nn_blocks, visualize_nn_graph,
    visualize_nn_training, visualize_nn_hyperparameters, visualize_nn_loss_landscape,
    visualize_nn_backpropagation, visualize_nn_weights, explain_nn_prediction,
]
for function in public_figures:
    print(f"{function.__name__}{inspect.signature(function)}")"""
            ),
            markdown(
                """
# 1. Regresión lineal

`visualize_lr` selecciona la geometría desde las variables originales: con 1 variable muestra un trazo 2D, que será recta para un modelo afín y curva si un `Pipeline` crea términos polinómicos; con 2 variables muestra una superficie 3D; con 3 o más usa una expansión/matriz matemática. Parámetros exclusivos:

### `visualize_lr`: datos, historia y evaluación

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `trained_estimator` | estimador ajustado | Fuente de coeficientes y predicciones. Acepta estimadores compatibles y `Pipeline`; la figura no vuelve a entrenar el objeto final. |
| `X`, `y` | matrices obligatorias | Datos usados para reconstruir/representar la trayectoria y calcular métricas. Las columnas originales de `X` deciden la familia visual: trazo 1D, superficie 2D o resumen N-D. Una expansión polinómica no convierte una entrada original 1D en la vista matricial: deforma su trazo. |
| `steps` | `60`, entero `>=1` | Cantidad de estados solicitados al reconstruir la historia. Más pasos suavizan la evolución, pero pueden ser submuestreados por `max_frames`. |
| `mode` | `'auto'`; `'iterative'`, `'final_interp'` | Decide cómo obtener la historia. `'auto'` reconstruye un replay si el estimador es incremental; `'iterative'` lo exige; `'final_interp'` interpola sintéticamente desde la línea base hasta el modelo ajustado. El replay tampoco afirma recuperar la llamada privada original a `fit`. |
| `baseline` | `'mean'`; también `'zeros'` | Punto inicial de la interpolación: predicción media con pendiente cero o todos los parámetros en cero. Cambia el recorrido, nunca el estado final. |
| `display_space` | `'original'` o `'scaled'` | En estimadores con transformación, muestra coeficientes/ejes en unidades originales o en el espacio transformado. Es especialmente visible con `StandardScaler`. |
| `show_history_context` | `True` | Muestra el subtítulo que declara si la trayectoria es replay real o interpolación sintética. Desactívelo sólo si ese contexto ya está explicado fuera de la figura. |
| `show_loss` | `True` | Añade la curva/panel temporal de pérdida. En 1D cambia de forma importante la composición disponible. |
| `metrics` | `None`; lista o diccionario | `None` usa la selección de la vista. Una lista admite `loss`, `mse`, `mae`, `r2`/`r_2`; un diccionario `{nombre: callable}` permite métricas propias. Controla tarjetas y series visibles. |
| `strict_loss` | `False` | Si es `True`, un error al calcular la pérdida detiene la figura; si es `False`, conserva la visualización y degrada ese panel. Útil para auditar integraciones. |
| `smooth` | `'ema'` o `None` | Aplica una media móvil exponencial sólo a la pérdida de un replay reconstruido o deja valores crudos. En interpolación sintética se conserva la evaluación sin EMA para mantener exacto el extremo final. No modifica el modelo ni las métricas crudas almacenadas. |
| `smooth_beta` | `0.85`, en `[0,1)` | Memoria de la EMA: cerca de 0 sigue los datos; cerca de 1 produce una curva más estable y retrasada. Sólo tiene efecto con `smooth='ema'` y una historia de tipo replay. |

### `visualize_lr`: tiempo, matemática y selección visible

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `frame_duration` | `80`, milisegundos `>=0` | Tiempo que permanece cada estado. Cambia velocidad de reproducción, no cuántos estados existen. |
| `transition_duration` | `None` o milisegundos `>=0` | Duración de la transición Plotly entre estados; `None` deja que la vista elija un valor coherente. |
| `max_frames` | `60`, entero o `None` | Tope de estados renderizados. Al reducirlo se seleccionan estados representativos, conservando primero y último; `None` delega el muestreo a `frame_step`. |
| `frame_step` | `10`, entero `>=1` | Salto entre estados cuando no se fija `max_frames`; valores altos aligeran HTML y hacen la evolución más discreta. |
| `animation_mode` | `'auto'`, `'native'`, `'hybrid'` | En 1D elige animación Plotly nativa o una interpolación híbrida más fluida; `'auto'` selecciona según el caso. En 2D/N-D no cambia de motor. |
| `fps` | `None` o número positivo | Objetivo de cuadros por segundo del modo híbrido 1D. Ajusta la cadencia; no crea estados del entrenamiento. |
| `interpolation_frames` | `3`, entero `>=1` | Intervalos visuales entre dos estados en modo híbrido 1D. Valores mayores hacen más fluida la deformación sin inventar nuevos estados semánticos. |
| `dec` | `4`, entero `>=0` | Número de decimales en ecuaciones, coeficientes, predicción y métricas. No altera ningún cálculo. |
| `show_objective` | `'auto'` o booleano | Muestra/oculta la función objetivo. `'auto'` la incorpora cuando la densidad y el espacio lo permiten. |
| `show_regularization` | `'auto'` o booleano | Muestra el término de regularización detectado, por ejemplo Ridge/Lasso. `True` es útil para comprobar que un `Pipeline` se interpreta correctamente. |
| `feature_names` | `None` o secuencia de textos | Nombra únicamente los ejes de entrada según las columnas originales de `X`. La matemática mantiene símbolos canónicos (`x₁`, `x₂`, …); en un `Pipeline` polinómico muestra `x₁²`, `x₁x₂`, `x₂³`, etc., aunque el nombre descriptivo del eje sea largo. |
| `sample_index` | `None` o índice válido | Elige la fila destacada en la derivación/predicción. `None` usa una muestra representativa; pruebe extremos para forzar ecuaciones largas. |
| `detail` | `'essential'`, `'academic'`, `'complete'` | Controla cuánto contexto matemático y diagnóstico aparece; compárelo con la tabla visual compartida. |

### `explain_lr_prediction`: explicación local de una predicción

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `trained_estimator`, `X_train`, `y_train` | obligatorios | Definen el modelo y el contexto de entrenamiento contra el que se explica una consulta. |
| `x_query` | `None` o una observación | Punto cuya predicción se desarrolla. Si falta, se escoge una fila de entrenamiento. |
| `yhat` | `None` o valor | Predicción suministrada externamente. Su uso depende de `prediction_source`. |
| `prediction_source` | `'model'`, `'provided'`, `'validate'` | `'model'` calcula con el estimador; `'provided'` usa `yhat`; `'validate'` compara `yhat` con el modelo y falla si no coincide dentro de las tolerancias. |
| `validation_rtol`, `validation_atol` | `1e-7`, `1e-9` | Tolerancias relativa y absoluta de la validación numérica. Súbalas sólo si hay diferencias esperadas de precisión. |
| `grid_points` | `250`, entero | Resolución de la recta/contexto 1D. Más puntos producen un trazo más denso y un HTML algo mayor. |
| `grid_2d_points` | `40`, entero | Resolución por eje de la superficie 2D; el coste crece aproximadamente con su cuadrado. |
| `display_space` | `'original'` o `'scaled'` | Explica entradas y coeficientes antes o después del preprocesamiento. |
| `dec` | `4`, entero `>=0` | Precisión textual de la sustitución numérica. |
| Parámetros visuales | tabla compartida | `title`, `theme`, `format`, `density`, `size`, `width`, `height`, `responsive` y `reduced_motion` sólo alteran la presentación. |
"""
            ),
            code(
                """diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
print(diabetes.DESCR.splitlines()[0])
print("X:", X_diabetes.shape, "y:", y_diabetes.shape)
print("Variables:", list(diabetes.feature_names))"""
            ),
            *case(
                "REAL-LR-1D-DIABETES",
                "recta, residuos, ecuación y animación híbrida con BMI del conjunto Diabetes",
                """X = X_diabetes[:, [2]]
lr_1d = LinearRegression().fit(X, y_diabetes)
display(visualize_lr(
    lr_1d, X, y_diabetes, steps=16, max_frames=8,
    animation_mode='hybrid', interpolation_frames=4, fps=35,
    metrics=['mse', 'mae', 'r2'], detail='academic',
    feature_names=['BMI estandarizado'], theme='academic', size='default',
))""",
            ),
            *case(
                "REAL-LR-2D-DIABETES",
                "cambio de recta a plano 3D usando BMI y presión arterial",
                """X = X_diabetes[:, [2, 3]]
lr_2d = LinearRegression().fit(X, y_diabetes)
display(visualize_lr(
    lr_2d, X, y_diabetes, steps=14, max_frames=7, baseline='zeros',
    show_loss=True, detail='academic', format='dashboard', size='default',
    feature_names=['BMI estandarizado', 'Presión arterial estandarizada'], sample_index=25,
))""",
            ),
            *case(
                "REAL-LR-ND-DIABETES",
                "límite N-dimensional: matriz de coeficientes, escalado, regularización y nombres reales",
                """lr_nd = make_pipeline(StandardScaler(), Ridge(alpha=25.0)).fit(X_diabetes, y_diabetes)
display(visualize_lr(
    lr_nd, X_diabetes, y_diabetes, steps=18, max_frames=6,
    display_space='original', metrics=['mse', 'mae', 'r2'], dec=5,
    detail='complete', show_objective=True, show_regularization=True,
    feature_names=diabetes.feature_names, sample_index=len(y_diabetes)-1,
    theme='academic', format='compact', size='wide',
))""",
            ),
            markdown(
                """
### Regresión polinómica: qué cambia en la figura

`PolynomialFeatures` transforma una variable original en varias columnas de
modelo. Para grado 3 y una entrada llamada `BMI`, el estimador recibe
`BMI`, `BMI²` y `BMI³`. La figura continúa siendo 1D porque sólo existe una
variable original, pero el trazo se deforma y deja de ser una recta. `BMI` es
una etiqueta descriptiva del eje; no sustituye el símbolo matemático.

La banda matemática superior ahora muestra dos niveles sincronizados:

- la forma simbólica
  `ŷ = θ₀ + θ₁·x₁ + θ₂·x₁² + θ₃·x₁³`;
- la sustitución numérica de todos esos coeficientes en cada estado.

Si hay `StandardScaler` después de la expansión, el encabezado declara
**transformed feature space**: los símbolos siguen siendo `x₁`, `x₁²` y
`x₁³`, mientras los valores y coeficientes pertenecen a las columnas
polinómicas escaladas. Para
grados mayores que cinco, la ecuación abrevia los términos centrales con
`…`; el cálculo y la curva siguen utilizando todos los coeficientes.

Con `Ridge` o `LinearRegression`, `mode='auto'` produce una interpolación
sintética desde `baseline` hasta la curva ajustada. `steps` crea estados
semánticos, `max_frames` decide cuántos se conservan e `interpolation_frames`
suaviza visualmente la deformación híbrida. Para observar un replay
reconstruido use un estimador incremental, como `SGDRegressor`, con
`mode='iterative'`.
"""
            ),
            *case(
                "REAL-LR-NONLINEAR-DIABETES",
                "regresión polinómica: eje BMI y ecuaciones animadas con x₁, x₁² y x₁³",
                """X = X_diabetes[:, [2]]
poly_lr = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    StandardScaler(),
    Ridge(alpha=4.0),
).fit(X, y_diabetes)
display(visualize_lr(
    poly_lr, X, y_diabetes, steps=24, max_frames=12,
    mode='final_interp', animation_mode='hybrid', interpolation_frames=4, fps=35,
    metrics=['mse', 'mae', 'r2'], detail='complete',
    feature_names=['BMI'], title='Regresión polinómica cúbica sobre BMI',
    theme='academic', size='wide',
))""",
            ),
            markdown(
                """
### Límites que cambian la lectura

- Con una variable hay geometría directa y animación `hybrid`; al pasar a dos aparece un plano 3D y sólo se admite animación nativa.
- A partir de tres variables no se inventa una proyección: la figura cambia a álgebra multivariable.
- Escalas muy distintas hacen ilegible una sustitución ingenua; un `Pipeline` y `display_space` permiten distinguir parámetros escalados y originales.
- Una transformación polinómica cubre no linealidad **en x**, aunque el estimador siga siendo lineal en sus coeficientes. Las potencias se muestran explícitamente en ecuaciones y sustituciones; una interpolación de `Ridge` no debe confundirse con pasos privados de entrenamiento. La librería no es un visualizador genérico de árboles, kernels o splines.
"""
            ),
            *case(
                "REAL-LR-PREDICTIONS",
                "todas las geometrías de explicación de predicción: 1D, 2D y N-D, incluida extrapolación",
                """display(explain_lr_prediction(
    lr_1d, X_diabetes[:, [2]], y_diabetes,
    x_query=[[X_diabetes[:, 2].max() + 0.04]], grid_points=180,
    theme='accessible',
))
display(explain_lr_prediction(
    lr_2d, X_diabetes[:, [2, 3]], y_diabetes,
    x_query=(X_diabetes[:, [2, 3]].max(axis=0) + 0.03).reshape(1, -1),
    grid_2d_points=28, size='wide',
))
display(explain_lr_prediction(
    lr_nd, X_diabetes, y_diabetes, x_query=X_diabetes[[40]],
    dec=5, size='wide',
))""",
            ),
            *case(
                "REAL-LR-VISUAL-PRESETS",
                "configuraciones visuales en sus extremos: dashboard accesible y reporte estático con tamaño explícito",
                """display(visualize_lr(
    lr_1d, X_diabetes[:, [2]], y_diabetes, steps=8, max_frames=5,
    show_history_context=False, theme='accessible',
    density='academic', responsive=True, size='notebook',
))
display(visualize_lr(
    lr_1d, X_diabetes[:, [2]], y_diabetes, steps=8, max_frames=5,
    theme='compact', format='report', width=900, height=620,
    reduced_motion=True, show_loss=False,
))""",
            ),
            markdown(
                """
# 2. Regresión logística

La forma depende de variables **y** clases: binaria 1D (sigmoide), binaria 2D (frontera/plano), binaria N-D (expansión), multiclase 1D (curvas), multiclase 2D (superficies/regiones) y multiclase N-D (matrices).

### `visualize_logistic`: datos, historia y evaluación

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `trained_estimator` | clasificador ajustado | Aporta clases, coeficientes y probabilidades. Puede ser un `Pipeline`; la figura respeta su preprocesamiento. |
| `X`, `y` | matrices obligatorias | Contexto de datos. El número de columnas y clases determina sigmoide, frontera/superficie o resumen matricial. |
| `steps` | `60`, entero `>=1` | Número de estados solicitados para replay/interpolación. No equivale a `max_iter` del clasificador. |
| `mode` | `'auto'`, `'iterative'`, `'final_interp'` | Selecciona replay incremental cuando es posible o interpolación hasta el modelo final. Fuerce modos para comprobar que el subtítulo describe honestamente la procedencia. |
| `baseline` | `'prior'` o `'uniform'` | Estado inicial: probabilidades según frecuencia de clases o equiprobables. Cambia la trayectoria y la pérdida inicial, no el clasificador final. |
| `display_space` | `'original'` o `'scaled'` | Decide si ejes, coeficientes y consulta se expresan en unidades del dato o del transformador. |
| `show_history_context` | `True` | Conserva el subtítulo de replay/interpolación y progreso. Ocultarlo no cambia la animación. |
| `show_loss` | `False` | Añade la evolución de log-loss. Active/desactive para probar composiciones con y sin panel temporal. |
| `metrics` | `None`; lista o diccionario | Admite `loss`, `log_loss`/`logloss`, `accuracy`, `f1`/`f1_score`, o `{nombre: callable}`. Controla tarjetas/series; no reentrena. |
| `strict_loss` | `False` | Convierte problemas al evaluar la pérdida en error (`True`) o permite una degradación visual (`False`). |
| `smooth` | `'ema'` o `None` | Suaviza las curvas de historia o muestra cada valor sin filtrar. |
| `smooth_beta` | `0.85`, en `[0,1)` | Intensidad/memoria del suavizado EMA; valores altos suavizan más y reaccionan más lento. |

### `visualize_logistic`: decisión, multiclase, tiempo y matemática

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `threshold` | `0.5`, estrictamente en `(0,1)` | Umbral de decisión binaria. Desplaza el punto/frontera donde cambia la clase; no cambia la curva de probabilidad. El eje conserva sólo el valor numérico y la geometría recibe una etiqueta corta `τ=…`, evitando que “threshold” invada el título de probabilidad. |
| `class_focus` | `None`, etiqueta o índice | En multiclase selecciona qué clase enfatizar en superficies y derivaciones. `None` deja que la vista elija; la etiqueta debe pertenecer a `classes_`. |
| `multiclass_link` | `'auto'`, `'softmax'`, `'ovr'` | Define cómo interpretar logits multiclase: normalización conjunta o una-vs-rest. `'auto'` intenta inferir la semántica del estimador. |
| `max_theta_cols` | `5`, entero `>=1` | Limita columnas de coeficientes mostradas en vistas N-D/multiclase. Es truncado visual: no elimina variables del cálculo. |
| `show_class_labels` | `False` | Sustituye índices por etiquetas reales en leyendas, ejes, probabilidades y ecuaciones cuando están disponibles. |
| `frame_duration` | `80`, milisegundos `>=0` | Tiempo de permanencia de cada estado de la animación. |
| `transition_duration` | `None` o milisegundos `>=0` | Duración de interpolación visual entre estados. |
| `max_frames` | `60`, entero o `None` | Tope de cuadros renderizados; conserva extremos al submuestrear. `None` permite gobernar con `frame_step`. |
| `frame_step` | `10`, entero `>=1` | Salto entre estados cuando no se aplica un tope explícito. |
| `dec` | `4`, entero `>=0` | Precisión textual de probabilidades, coeficientes y métricas. |
| `show_objective` | `'auto'` o booleano | Controla la ecuación de log-loss/objetivo según espacio y densidad. |
| `show_regularization` | `'auto'` o booleano | Muestra el término de penalización detectado; fuerce `True` para comparar valores de `C`. |
| `feature_names` | `None` o secuencia de textos | Nombra los ejes de entrada respetando el orden de `X`; las ecuaciones conservan `x₁`, `x₂`, … y sus transformaciones. |
| `sample_index` | `None` o índice válido | Escoge la observación usada en la derivación puntual. Use muestras cercanas y lejanas a la frontera para comparar. |
| `detail` | `'essential'`, `'academic'`, `'complete'` | Incrementa el contenido matemático/diagnóstico visible sin cambiar probabilidades. |

### `explain_logistic_prediction`: explicación local

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `trained_estimator`, `X_train`, `y_train` | obligatorios | Proporcionan modelo, clases y contexto de referencia. |
| `x_query` | `None` o una observación | Punto cuya probabilidad y clase se descomponen; `None` escoge una observación de entrenamiento. |
| `p_hat` | `None`, probabilidad o vector | Probabilidad suministrada para explicaciones desacopladas del estimador. Debe respetar el orden de clases. |
| `y_hat` | `None` o etiqueta | Clase predicha suministrada; se usa o valida según `prediction_source`. |
| `prediction_source` | `'model'`, `'provided'`, `'validate'` | Calcula con el modelo, confía en `p_hat`/`y_hat`, o compara ambos caminos y falla ante una discrepancia. |
| `validation_rtol`, `validation_atol` | `1e-7`, `1e-9` | Tolerancias relativa/absoluta al validar probabilidades numéricas. |
| `grid_points` | `250`, entero | Resolución del dominio 1D sobre el que se dibujan probabilidades y frontera. |
| `show_class_labels` | `False` | Usa nombres reales de clases en vez de índices. |
| `display_space` | `'original'` o `'scaled'` | Expresa la consulta y los coeficientes antes o después del preprocesamiento. |
| `multiclass_link` | `'auto'`, `'softmax'`, `'ovr'` | Elige la transformación de puntuaciones a probabilidades para la derivación multiclase. |
| `dec` | `4`, entero `>=0` | Cantidad de decimales en logits, probabilidades y sustituciones. |
| Parámetros visuales | tabla compartida | `title`, `theme`, `format`, `density`, `size`, `width`, `height`, `responsive` y `reduced_motion` afectan sólo la presentación. |
"""
            ),
            code(
                """cancer = load_breast_cancer()
iris = load_iris()
wine = load_wine()
print("Breast cancer:", cancer.data.shape, list(cancer.target_names))
print("Iris:", iris.data.shape, list(iris.target_names))
print("Wine:", wine.data.shape, list(wine.target_names))"""
            ),
            markdown(
                """
### Una sola clase: frontera matemática del modelo

Regresión logística estima probabilidades **entre clases**. Con una única clase no existe frontera ni función de pérdida identificable; scikit-learn rechaza el ajuste y Mlektic no debe fabricar una figura. Esta celda documenta el límite y continúa el laboratorio.
"""
            ),
            code(
                """X_one_class = cancer.data[cancer.target == 0, [0]]
y_one_class = np.repeat('malignant', len(X_one_class))
try:
    LogisticRegression().fit(X_one_class, y_one_class)
except ValueError as error:
    print("Caso esperado, sin figura válida:", error)"""
            ),
            *case(
                "REAL-LOG-BINARY-1D",
                "sigmoide binaria 1D con clases semánticas y umbral alto",
                """X = cancer.data[:, [0]]
y = cancer.target_names[cancer.target]
log_1d = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)).fit(X, y)
display(visualize_logistic(
    log_1d, X, y, steps=16, max_frames=8, show_loss=True,
    threshold=0.72, show_class_labels=True, metrics=['log_loss', 'accuracy', 'f1'],
    detail='complete', feature_names=['radio medio'], theme='classroom',
))""",
            ),
            *case(
                "REAL-LOG-BINARY-2D",
                "frontera binaria 2D con umbral etiquetado dentro de la geometría, sin invadir el eje de probabilidad",
                """X = cancer.data[:, [0, 3]]
y = cancer.target
log_2d = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=1500)).fit(X, y)
display(visualize_logistic(
    log_2d, X, y, steps=14, max_frames=7, show_loss=True,
    display_space='original', threshold=0.35, detail='complete',
    feature_names=['radio medio', 'área media'], size='wide',
))""",
            ),
            *case(
                "REAL-LOG-BINARY-ND",
                "treinta variables: identidad sigmoide única, sesgo en una fila independiente y métricas alineadas",
                """log_nd = make_pipeline(StandardScaler(), LogisticRegression(C=0.15, max_iter=1500)).fit(cancer.data, cancer.target)
display(visualize_logistic(
    log_nd, cancer.data, cancer.target, steps=12, max_frames=6,
    show_loss=True, metrics=['log_loss', 'accuracy', 'f1'], max_theta_cols=4,
    detail='complete', show_regularization=True,
    feature_names=list(cancer.feature_names), sample_index=120,
    theme='academic', format='compact', size='wide',
))""",
            ),
            *case(
                "REAL-LOG-MULTI-1D",
                "tres curvas de Iris con definición vectorial alineada y derivaciones verticalmente separadas",
                """X = iris.data[:, [2]]
y = iris.target_names[iris.target]
iris_1d = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)).fit(X, y)
display(visualize_logistic(
    iris_1d, X, y, steps=14, max_frames=7, show_loss=True,
    show_class_labels=True, multiclass_link='softmax',
    feature_names=['longitud de pétalo'], theme='accessible',
))""",
            ),
            *case(
                "REAL-LOG-MULTI-2D",
                "tres superficies de Iris con la misma notación de una variable",
                """X = iris.data[:, [2, 3]]
y = iris.target_names[iris.target]
iris_2d = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)).fit(X, y)
display(visualize_logistic(
    iris_2d, X, y, steps=14, max_frames=7, show_loss=True,
    show_class_labels=True, multiclass_link='softmax',
    feature_names=['longitud de pétalo', 'ancho de pétalo'], theme='accessible',
))""",
            ),
            *case(
                "REAL-LOG-MULTI-ND",
                "trece variables y tres clases de vino: matrices coherentes y clases intermedias explícitamente abreviadas",
                """wine_labels = wine.target_names[wine.target]
wine_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.4, max_iter=1800)).fit(wine.data, wine_labels)
display(visualize_logistic(
    wine_model, wine.data, wine_labels, steps=14, max_frames=6,
    show_loss=True, show_class_labels=True, max_theta_cols=3,
    metrics=['log_loss', 'accuracy', 'f1'], detail='complete',
    feature_names=wine.feature_names, sample_index=75,
    theme='academic', size='wide',
))""",
            ),
            *case(
                "REAL-LOG-MULTI-ND-NOTEBOOK",
                "mismas trece variables en tamaño notebook: fracciones compactas y puntos suspensivos alineados",
                """display(visualize_logistic(
    wine_model, wine.data, wine_labels, steps=14, max_frames=6,
    show_loss=True, show_class_labels=True, max_theta_cols=3,
    metrics=['log_loss', 'accuracy', 'f1'], detail='complete',
    feature_names=wine.feature_names, sample_index=75,
    theme='academic', size='notebook',
))""",
            ),
            *case(
                "REAL-LOG-PREDICTIONS",
                "explicación local binaria 1D/2D y multiclase N-D",
                """display(explain_logistic_prediction(
    log_1d, cancer.data[:, [0]], cancer.target_names[cancer.target],
    x_query=[[18.0]], show_class_labels=True, grid_points=180,
    theme='classroom',
))
display(explain_logistic_prediction(
    log_2d, cancer.data[:, [0, 3]], cancer.target,
    x_query=cancer.data[[12]][:, [0, 3]], show_class_labels=True, size='wide',
))
display(explain_logistic_prediction(
    wine_model, wine.data, wine_labels, x_query=wine.data[[10]],
    show_class_labels=True, dec=5, size='wide', format='report',
))""",
            ),
            markdown(
                """
# 3. Redes neuronales artificiales (PyTorch)

Las figuras no dependen sólo del número de variables, sino de la **topología ejecutada** y de lo que se registró. `visualize_nn(..., view=...)` enruta a `architecture`, `blocks`, `hyperparameters`, `graph`, `training`, `weights`, `activations` o `backpropagation`.

### Router `visualize_nn`: entradas y selección de vista

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `model` | módulo PyTorch obligatorio | Red que se inspecciona. Se usa su topología y, según la vista, sus parámetros/ejecución; no se entrena dentro de la función. |
| `input_sample` | `None` o tensor/estructura de entrada | Permite ejecutar y capturar formas, activaciones y ramas reales. Es necesario en `graph`/`activations` y para arquitecturas dinámicas. |
| `history` | `None` o historial del recorder | Proporciona pasos, pérdidas, pesos, gradientes y activaciones. Las vistas animadas sólo pueden mostrar campos que fueron registrados. |
| `optimizer`, `loss_fn`, `scheduler` | `None` u objetos PyTorch | Enriquecen `view='hyperparameters'` con grupos, tasa de aprendizaje, penalizaciones, función de pérdida y planificador. |
| `view` | `'architecture'` | Enruta a `'architecture'`, `'blocks'`, `'hyperparameters'`, `'graph'`, `'training'`, `'weights'`, `'activations'` o `'backpropagation'`. Cambiarlo puede exigir `input_sample` o `history`. |
| `title` | `None` o texto | Sustituye el encabezado automático de la vista elegida. |
| `capture_backend` | `'auto'`, `'fx'`, `'hooks'` | Motor para descubrir el grafo ejecutado. FX ofrece estructura simbólica; hooks observan una ejecución real y suelen cubrir modelos no trazables; `'auto'` intenta el mejor camino. |
| `input_kwargs` | `None` o diccionario | Argumentos nombrados que se pasan al `forward` junto con `input_sample`; imprescindible en modelos con múltiples entradas o parámetros keyword-only. |
| `max_nodes` | `48`, entero positivo | Tope de operaciones/bloques dibujados en grafos capturados. Trunca la representación, no la ejecución de la red. |
| `show_formulas` | `True` | Incluye ecuaciones/resúmenes de cada operación cuando existen; desactivarlo libera espacio en grafos largos. |

### Grafo denso y evolución: `visualize_nn_graph`

Los siguientes argumentos también se aceptan en el router y se usan cuando `view='graph'`.

| Parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `model`, `input_sample`, `history` | obligatorios en la función directa | Combinan topología, activación de una muestra y estados temporales. |
| `max_neurons` | `8`, entero positivo | Número máximo de neuronas visibles por capa; las restantes se resumen. No poda ni modifica la red. |
| `max_frames` | `20`, entero o `None` | Limita estados del historial y conserva una muestra temporal representativa; `None` usa todos. |
| `frame_duration` | `180` ms (router: `120`) | Cadencia de reproducción de los estados registrados. |
| `node_color_mode` | `'value'`, `'relative'` | Colorea nodos por magnitud/valor absoluto o normalizado respecto al conjunto visible. Compare ambos cuando las capas tienen escalas distintas. |
| `edge_color_mode` | `'weight'`, `'signal'` | Codifica en aristas el peso del parámetro o la señal resultante peso×activación. `'signal'` depende también de la muestra. |
| `evolution_mode` | `'absolute'`, `'updates'`, `'hybrid'` | Muestra valores absolutos, cambios entre estados o ambos. `'updates'` sirve para ver qué aprendió, aunque los pesos absolutos sean grandes. |
| `update_reference` | `'previous'`, `'initial'` | Calcula el cambio contra el cuadro anterior o contra el primero; sólo afecta modos con actualizaciones. |
| `update_scale` | `'global'`, `'frame'` | Mantiene una escala de color comparable en toda la animación o reescala cada cuadro para revelar cambios pequeños. |
| `show_update_panel` | `None` o booleano | Fuerza/oculta el resumen de actualizaciones; `None` lo decide a partir de `evolution_mode` y espacio. |
| `show_loss_panel` | `False` | Añade la pérdida alineada con el cuadro activo. Requiere que el historial contenga `loss`. |
| `show_backpropagation` | `False` | Superpone información de gradiente/flujo hacia atrás cuando fue registrada. |
| `top_k_updates` | `None` o entero positivo | Reduce el panel a los K cambios de mayor magnitud; no altera las aristas del modelo. |
| `interpolation_frames` | `0`, entero `>=0` | Inserta cuadros visuales entre estados registrados para suavizar el movimiento; no inventa nuevas actualizaciones de entrenamiento. |
| `capture_backend`, `input_kwargs`, `max_nodes`, `show_formulas` | valores del router | Controlan la captura del grafo ejecutado antes de construir la vista densa. |

### Arquitectura, bloques e inspección

| Función / parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `visualize_nn_architecture(model, input_sample)` | `input_sample=None` | Dibuja una síntesis por capas; con muestra puede inferir formas y arquitectura ejecutada. |
| `history` | `None` | Añade contexto registrado si está disponible, sin ser obligatorio para una arquitectura estática. |
| `max_layers` | `8`, entero positivo | Limita capas visibles y resume el resto. Pruebe una red profunda para comprobar la elipsis. |
| `architecture_mode` | `'legacy'`, `'blocks'` | `'legacy'` usa la vista pedagógica por capas; `'blocks'` usa el grafo de operaciones capturado, mejor para ramas/CNN/RNN. |
| `capture_backend`, `input_kwargs`, `max_nodes`, `show_formulas` | como en el router | Se aplican principalmente a `architecture_mode='blocks'`. |
| `inspect_nn(model, inputs)` | devuelve `NeuralGraph` | Inspección sin figura para revisar nodos, aristas, formas y procedencia antes de visualizar. |
| `backend` | `'auto'`, `'fx'`, `'hooks'` | Equivalente a `capture_backend`, pero con el nombre propio de `inspect_nn`/`visualize_nn_blocks`. |
| `visualize_nn_blocks(model, inputs)` | entradas obligatorias | Dibuja directamente el grafo de bloques ejecutados; acepta `input_kwargs`, `backend`, `title`, `max_nodes` y `show_formulas`. |

### Vistas dedicadas de entrenamiento

| Función / parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `visualize_nn_training(history)` | historial obligatorio | Anima pérdida y métricas alineadas con los pasos. |
| `max_metrics` | `3`, entero `<=3` | Tope de series de métricas para conservar legibilidad; el historial no se recorta. |
| `max_frames`, `frame_duration` | `30`, `120` ms | Submuestreo temporal y velocidad de reproducción. |
| `visualize_nn_hyperparameters(model)` | modelo obligatorio | Resume arquitectura y configuración de optimización; acepta `history`, `optimizer`, `loss_fn` y `scheduler`, cada uno opcional. |
| `history` | `None` | Añade metadatos observados del entrenamiento y permite contrastarlos con los objetos actuales. |
| `optimizer` | `None` | Muestra algoritmo, grupos, learning rate, momentum/betas y regularización disponibles. |
| `loss_fn` | `None` | Identifica la función objetivo y sus opciones relevantes. |
| `scheduler` | `None` | Expone la estrategia que cambia la tasa de aprendizaje. |
| `visualize_nn_weights(history)` | historial con pesos | Muestra matrices/vectores de parámetros a través del tiempo. |
| `parameter` | `None` o nombre exacto | Selecciona un tensor, por ejemplo `'0.weight'`; `None` deja que la vista elija/resuma varios. |
| `max_rows`, `max_cols` | `4`, `5` | Recortan la matriz visible para evitar celdas ilegibles; no cambian el tensor registrado. |
| `max_parameters` | `6` | Tope de tensores mostrados cuando no se selecciona uno. |
| `max_frames`, `frame_duration` | `30`, `120` ms | Cantidad de estados visibles y velocidad. |
| Activaciones (`view='activations'`) | `input_sample`, `history` | Reproduce resúmenes/vectores capturados por capa. Requiere `capture_activations=True`; usa `max_frames` y `frame_duration`. |
| `visualize_nn_backpropagation(model, history)` | modelo e historial obligatorios | Visualiza magnitudes/flujo de gradiente por capa; requiere gradientes registrados. |
| `input_sample` | `None` | Añade contexto de activación para la muestra; puede omitirse si el historial basta. |
| `max_layers` | `8` | Limita capas visibles en redes profundas. |
| `max_frames`, `frame_duration` | `20`, `900` ms | Submuestreo y pausa por fase; la duración alta facilita seguir el recorrido hacia atrás. |

### Paisaje de pérdida y explicación de una predicción

| Función / parámetro | Predeterminado / opciones | Qué hace y qué debería observar |
|---|---|---|
| `visualize_nn_loss_landscape(model, inputs, targets, loss_fn, history)` | cinco argumentos obligatorios | Evalúa un corte 2D reproducible alrededor de los parámetros y superpone la trayectoria. Es una proyección, no todo el espacio de parámetros. |
| `grid_size` | `23`, entero entre `9` y `61` | Resolución por eje del paisaje; el coste crece aproximadamente con el cuadrado. |
| `span` | `None` o número `>0` | Radio del corte alrededor del punto de referencia; `None` estima una escala a partir de la trayectoria. |
| `max_frames`, `frame_duration` | `20`, `180` ms | Puntos temporales de la trayectoria que se animan y su cadencia. |
| `seed` | `17`, entero | Fija las dos direcciones aleatorias del corte para poder comparar configuraciones. |
| `explain_nn_prediction(model, x_query)` | modelo y consulta obligatorios | Desarrolla capas, operaciones y salida para una muestra concreta. |
| `history` | `None` | Permite explicar estados registrados cuando `parameter_state='training_replay'`. |
| `dec` | `4`, entero `>=0` | Precisión de activaciones, pesos y resultados mostrados. |
| `max_layers_math` | `6`, entero positivo | Límite de capas desarrolladas algebraicamente; las restantes se resumen. |
| `max_neurons_math` | `8`, entero positivo | Límite de neuronas/términos por capa en las ecuaciones. |
| `max_frames`, `frame_duration` | `12`, `220` ms | Estados explicados y velocidad cuando se reproduce historia. |
| `parameter_state` | `'final'`, `'training_replay'` | Usa los parámetros actuales o reconstruye los registrados. Es clave para no confundir la predicción final con una etapa temprana. |

Todas las figuras de redes aceptan `theme`, `format`, `density`, `size`, `width`, `height`, `responsive` y `reduced_motion`. `math_font_scale` admite valores entre `0.75` y `2.0` y escala solamente ecuaciones/anotaciones matemáticas. Los límites `max_*` cambian sólo la representación visible: nunca cambian la red.
"""
            ),
            markdown(
                """
## Registro reproducible de una red simple sobre Iris

`TorchTrainingRecorder` no cambia el entrenamiento: toma instantáneas compactas para que las figuras dispongan de una historia real. Llame `record` después de `optimizer.step()` y antes de borrar gradientes si quiere conservar tanto el parámetro actualizado como el gradiente que lo produjo.

| Parámetro del constructor | Predeterminado | Qué registra / para qué sirve |
|---|---|---|
| `model` | obligatorio | Red observada; los hooks se conectan a sus módulos hoja. |
| `optimizer` | `None` | Grupos e hiperparámetros; necesario para enriquecer la vista de hiperparámetros/estado. |
| `loss_fn` | `None` | Identidad y configuración de la función objetivo. |
| `scheduler` | `None` | Estado/configuración del planificador de learning rate. |
| `record_every` | `1`, entero `>=1` | Guarda uno de cada N llamados a `record`; reduce memoria y resolución temporal. |
| `capture_weights` | `True` | Copia tensores de parámetros para `weights`, grafo evolutivo y replay. |
| `capture_gradients` | `True` | Copia gradientes para backpropagación y actualizaciones. Debe registrarse antes de `zero_grad()`. |
| `capture_activations` | `True` | Instala hooks y guarda estadísticas/vectores para la vista de activaciones; desactívelo para reducir coste. |
| `capture_buffers` | `True` | Guarda buffers como medias/varianzas de BatchNorm, útiles en redes con estado no entrenable. |
| `capture_optimizer_state` | `False` | Conserva resúmenes del estado interno del optimizador; aporta detalle, pero consume más memoria. |
| `max_tensor_elements` | `4096`, entero `>=1` | No copia completo un tensor mayor que el límite; mantiene sus normas/resúmenes para que el registro sea acotado. |
| `max_activation_elements` | `512`, entero `>=1` | Límite del vector de activación conservado por capa; siempre se guardan estadísticas compactas. |

`record(step, ...)` acepta `loss`, `metrics`, `predictions`, `targets`, `task={'auto','classification','regression'}`, `capture_phase` y `observation_phase`. Con predicciones y objetivos puede inferir métricas; las fases documentan si una observación corresponde a antes o después de la actualización. Para pesos/backpropagación se requieren las capturas correspondientes; para activaciones, `capture_activations=True`.
"""
            ),
            code(
                """import torch

torch.manual_seed(SEED)
iris_X_np = StandardScaler().fit_transform(iris.data).astype('float32')
iris_y_np = iris.target.astype('int64')
iris_X_t = torch.tensor(iris_X_np)
iris_y_t = torch.tensor(iris_y_np)

iris_nn = torch.nn.Sequential(
    torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3)
)
iris_opt = torch.optim.Adam(iris_nn.parameters(), lr=0.04, weight_decay=0.001)
iris_loss = torch.nn.CrossEntropyLoss()
iris_sched = torch.optim.lr_scheduler.StepLR(iris_opt, step_size=4, gamma=0.7)
iris_rec = TorchTrainingRecorder(
    iris_nn, optimizer=iris_opt, loss_fn=iris_loss, scheduler=iris_sched,
    capture_weights=True, capture_gradients=True, capture_activations=True,
    max_tensor_elements=4096, max_activation_elements=128,
)
for step in range(1, 9):
    iris_opt.zero_grad()
    logits = iris_nn(iris_X_t)
    loss = iris_loss(logits, iris_y_t)
    loss.backward()
    iris_opt.step()
    iris_sched.step()
    iris_rec.record(step, loss=loss, predictions=logits, targets=iris_y_t, task='classification')
iris_rec.close()
iris_history = iris_rec.to_history()
print("Frames registrados:", iris_history['steps'])"""
            ),
            *case(
                "REAL-NN-SIMPLE-ALL-VIEWS-A",
                "arquitectura, grafo de bloques e hiperparámetros de la red Iris",
                """display(visualize_nn_architecture(
    iris_nn, iris_X_t[:1], history=iris_history,
    max_layers=8, architecture_mode='legacy', theme='classroom',
))
display(visualize_nn_architecture(
    iris_nn, iris_X_t[:1], architecture_mode='blocks', capture_backend='auto',
    max_nodes=16, show_formulas=True, theme='academic', size='wide',
))
graph_ir = inspect_nn(iris_nn, iris_X_t[:1], backend='auto')
print("Grafo capturado:", len(graph_ir.nodes), "nodos y", len(graph_ir.edges), "aristas")
display(visualize_nn_hyperparameters(
    iris_nn, history=iris_history, optimizer=iris_opt,
    loss_fn=iris_loss, scheduler=iris_sched,
    theme='academic', size='wide',
))""",
            ),
            *case(
                "REAL-NN-SIMPLE-ALL-VIEWS-B",
                "entrenamiento, pesos, activaciones y retropropagación registrados",
                """display(visualize_nn_training(
    iris_history, max_frames=None, max_metrics=3,
    frame_duration=220, theme='academic',
))
display(visualize_nn_weights(
    iris_history, parameter='0.weight', max_rows=4, max_cols=4,
    max_parameters=4, max_frames=8, theme='academic', size='wide',
))
display(visualize_nn(
    iris_nn, iris_X_t[:1], history=iris_history, view='activations',
    max_frames=8, frame_duration=220, theme='accessible', size='wide',
))
display(visualize_nn_backpropagation(
    iris_nn, iris_history, input_sample=iris_X_t[:1],
    max_layers=6, max_frames=8, frame_duration=650,
    theme='classroom', math_font_scale=1.15, size='wide',
))""",
            ),
            *case(
                "REAL-NN-GRAPH-MODES",
                "modos absolutos y de actualización en el grafo denso animado",
                """display(visualize_nn_graph(
    iris_nn, iris_X_t[12], iris_history,
    max_neurons=8, max_frames=8, frame_duration=260,
    node_color_mode='value', edge_color_mode='weight', evolution_mode='absolute',
    show_loss_panel=True, theme='academic', size='wide',
))
display(visualize_nn_graph(
    iris_nn, iris_X_t[12], iris_history,
    max_neurons=6, max_frames=6, frame_duration=300,
    node_color_mode='relative', edge_color_mode='signal', evolution_mode='hybrid',
    update_reference='initial', update_scale='frame', show_update_panel=True,
    show_backpropagation=True, top_k_updates=6, interpolation_frames=2,
    theme='accessible', size='wide',
))""",
            ),
            *case(
                "REAL-NN-PREDICTION-LANDSCAPE",
                "sustitución de una predicción final, replay y corte 2D del paisaje de pérdida",
                """display(explain_nn_prediction(
    iris_nn, iris_X_t[12], history=iris_history,
    parameter_state='final', dec=4, max_layers_math=5, max_neurons_math=6,
    theme='academic', size='wide',
))
display(explain_nn_prediction(
    iris_nn, iris_X_t[12], history=iris_history,
    parameter_state='training_replay', max_frames=6, frame_duration=260,
    theme='classroom', size='wide',
))
display(visualize_nn_loss_landscape(
    iris_nn, iris_X_t, iris_y_t, iris_loss, iris_history,
    grid_size=9, span=None, max_frames=5, frame_duration=280, seed=SEED,
    theme='academic', size='wide',
))""",
            ),
            markdown(
                """
## Red densa más compleja sobre Digits

Digits contiene imágenes reales de dígitos manuscritos de 8×8. Primero las tratamos como 64 variables para forzar truncamientos representativos (`max_layers`, `max_nodes`, `max_neurons`, `max_parameters`) sin ocultar que la red real es mayor.
"""
            ),
            code(
                """digits = load_digits()
digits_X_t = torch.tensor((digits.data / 16.0).astype('float32'))
digits_y_t = torch.tensor(digits.target.astype('int64'))

torch.manual_seed(SEED)
digits_dense = torch.nn.Sequential(
    torch.nn.Linear(64, 48), torch.nn.ReLU(),
    torch.nn.Linear(48, 24), torch.nn.ReLU(),
    torch.nn.Linear(24, 12), torch.nn.ReLU(),
    torch.nn.Linear(12, 10),
)
digits_opt = torch.optim.Adam(digits_dense.parameters(), lr=0.015)
digits_loss = torch.nn.CrossEntropyLoss()
digits_rec = TorchTrainingRecorder(digits_dense, optimizer=digits_opt, loss_fn=digits_loss)
train_X, train_y = digits_X_t[:512], digits_y_t[:512]
for step in range(1, 7):
    digits_opt.zero_grad(); logits = digits_dense(train_X); loss = digits_loss(logits, train_y)
    loss.backward(); digits_opt.step()
    digits_rec.record(step, loss=loss, predictions=logits, targets=train_y, task='classification')
digits_rec.close(); digits_history = digits_rec.to_history()"""
            ),
            *case(
                "REAL-NN-COMPLEX-DENSE",
                "red profunda y ancha con límites visuales que resumen sin cambiar el modelo",
                """display(visualize_nn_architecture(
    digits_dense, digits_X_t[:1], history=digits_history,
    max_layers=6, theme='classroom', size='wide',
))
display(visualize_nn_blocks(
    digits_dense, digits_X_t[:1], max_nodes=7, show_formulas=False,
    backend='auto', theme='academic', size='wide',
))
display(visualize_nn_training(
    digits_history, max_frames=None, theme='academic', size='wide',
))
display(visualize_nn_weights(
    digits_history, max_rows=3, max_cols=4, max_parameters=3,
    max_frames=6, theme='academic', size='wide',
))""",
            ),
            markdown(
                """
## CNN, red recurrente y otras topologías

Para convoluciones, recurrencia, atención y ramas, una cuadrícula de neuronas densas sería engañosa. Mlektic captura el grafo ejecutado y muestra bloques, fórmulas y formas tensoriales. En estos casos `visualize_nn(..., view='graph')` puede enrutar deliberadamente a bloques y dejar constancia en `layout.meta`.
"""
            ),
            code(
                """class DigitsCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size=3), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(36, 10))
    def forward(self, x):
        return self.classifier(self.features(x))

class DigitsLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(8, 12, batch_first=True, bidirectional=True)
        self.output = torch.nn.Linear(24, 10)
    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.output(sequence[:, -1, :])

class ResidualClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(8, 8)
        self.output = torch.nn.Linear(8, 3)
    def forward(self, x):
        residual = x + torch.relu(self.hidden(x))
        return self.output(residual)

cnn = DigitsCNN()
lstm = DigitsLSTM()
residual = ResidualClassifier()
digits_images = digits_X_t.reshape(-1, 1, 8, 8)
digits_sequences = digits_X_t.reshape(-1, 8, 8)"""
            ),
            *case(
                "REAL-NN-CNN-RNN",
                "CNN y LSTM sobre Digits: operaciones y dimensiones tensoriales completas",
                """display(visualize_nn_architecture(
    cnn, digits_images[:1], architecture_mode='blocks',
    capture_backend='auto', max_nodes=20, show_formulas=True,
    theme='classroom', size='wide',
))
display(visualize_nn_blocks(
    lstm, digits_sequences[:2], backend='auto', max_nodes=20,
    show_formulas=True, theme='academic', size='wide',
))""",
            ),
            *case(
                "REAL-NN-BRANCH-ATTENTION",
                "rama residual y atención multi-cabeza como casos no secuenciales",
                """display(visualize_nn_blocks(
    residual, torch.randn(2, 8), backend='auto',
    theme='accessible', show_formulas=True, size='wide',
))
attention = torch.nn.MultiheadAttention(8, 2, dropout=0.1, batch_first=True)
q = torch.randn(2, 6, 8)
display(visualize_nn_blocks(
    attention, (q, q.clone(), q.clone()), backend='auto',
    theme='academic', max_nodes=24, show_formulas=True, size='wide',
))""",
            ),
            markdown(
                """
## Guía de elección

- **¿Cómo cambia el modelo durante el ajuste?** `training`, `weights`, `activations`, `backpropagation` y `graph`, siempre con historial registrado.
- **¿Qué cálculo ejecutó esta entrada?** `blocks`/`inspect_nn`; es la opción fiel para CNN, LSTM, atención y ramas.
- **¿Qué significa una predicción concreta?** `explain_*_prediction` o `explain_nn_prediction`.
- **¿Necesito una lámina?** `format='report'`, `reduced_motion=True`, tamaño explícito o `size='wide'`.
- **¿La figura se satura?** reduzca `max_frames`, `max_nodes`, `max_neurons`, `max_layers`, `max_parameters`, filas/columnas o use `class_focus`; estos controles resumen la vista, no el cálculo.

Con esto quedan ejercitadas todas las familias de figura públicas y todos los cambios estructurales por dimensionalidad, número de clases, topología e historial. Las tablas anteriores son la referencia de parámetros; las firmas impresas al inicio permiten comprobar los valores predeterminados de la versión instalada.
"""
            ),
        ]
    )
    notebook(
        "Escenarios realistas y referencia completa de figuras",
        "Laboratorio integral de regresión lineal, regresión logística y redes neuronales con datos reales, límites matemáticos y configuraciones visuales.",
        cells,
        "learn/learn_05_realistic_scenarios_with_neural.ipynb",
    )


def build_manifest() -> None:
    """Build the machine-readable case inventory after all notebooks exist."""
    cases = {}
    for path in sorted((NOTEBOOKS / "qa").glob("*.ipynb")) + sorted((NOTEBOOKS / "learn").glob("*.ipynb")):
        nb = nbf.read(path, as_version=4)
        for index, cell in enumerate(nb.cells):
            case_id = cell.get("metadata", {}).get("mlektic_case_id")
            if case_id:
                cases[case_id] = {"notebook": path.relative_to(ROOT).as_posix(), "cell_index": index}
    documents = {
        "codeasdoc/introduction.rst": ["SMOKE-LR-1D"],
        "codeasdoc/installation.rst": ["LEARN-GETTING-STARTED"],
        "codeasdoc/getting_started.rst": ["LEARN-GETTING-PREDICTION"],
        "codeasdoc/linear_lesson.rst": ["LEARN-LINEAR-PLANE", "LEARN-LINEAR-POLY"],
        "codeasdoc/logistic_lesson.rst": ["LEARN-LOGISTIC-BINARY", "LEARN-LOGISTIC-MULTI"],
        "codeasdoc/neural_lesson.rst": [
            "LEARN-NN-ARCHITECTURE",
            "LEARN-NN-TRAINING",
            "NN-ROUTER-HYPERPARAMETERS",
            "NN-GALLERY-FORWARD-SUBSTITUTION",
            "LEARN-NN-BRANCHES",
            "LEARN-NN-ATTENTION",
        ],
        "codeasdoc/neural_execution_graphs.rst": [
            "NN-ROUTER-ARCHITECTURE",
            "NN-ROUTER-BLOCKS",
            "NN-ROUTER-HYPERPARAMETERS",
            "NN-ROUTER-GRAPH",
            "NN-GRAPH-WITHOUT-BACKPROP",
            "NN-GRAPH-WITH-BACKPROP",
            "NN-ROUTER-BACKPROPAGATION",
            "NN-LOSS-LANDSCAPE-XOR",
            "NN-BACKPROP-XOR",
            "NN-GRAPH-HYBRID-UPDATES",
            "NN-GRAPH-CUMULATIVE-UPDATES",
            "NN-GRAPH-FRAME-NORMALIZED-UPDATES",
            "NN-ROUTER-TRAINING",
            "NN-ROUTER-WEIGHTS",
            "NN-ROUTER-ACTIVATIONS",
            "NN-TRAINING-QUERY-REPLAY",
            "NN-GALLERY-FORWARD-SUBSTITUTION",
            "NN-SYNTH-REG-TRAINING",
            "NN-REAL-IRIS-GRAPH",
            "NN-REAL-CANCER-PREDICTION",
            "NN-REAL-DIGITS-BLOCKS",
            "NN-REAL-DIGITS-DENSE-HEAD-GRAPH",
            "NN-BLOCK-RESIDUAL",
            "NN-BLOCK-SHARED",
            "NN-BLOCK-CONV",
            "NN-BLOCK-EMBEDDING",
            "NN-BLOCK-LSTM",
            "NN-BLOCK-ATTENTION",
            "NN-BLOCK-MULTI-IO",
            "NN-LARGE-HUNDREDS-MANY-LAYERS",
            "NN-LARGE-ARCHITECTURE",
            "NN-LARGE-BLOCKS",
            "NN-LARGE-GRAPH",
            "NN-LARGE-FORWARD-SUBSTITUTION",
            "NN-LARGE-TRAINING",
            "NN-LARGE-WEIGHTS",
            "NN-LARGE-ACTIVATIONS",
            "NN-LARGE-BACKPROPAGATION",
            "NN-LARGE-LOSS-LANDSCAPE",
        ],
        "codeasdoc/history_semantics.rst": ["MOTION-NATIVE-SLOW", "MOTION-HYBRID-FLUID"],
        "codeasdoc/mathematical_parity.rst": ["LR-PIPE-ORIGINAL", "LOG-BINARY-THRESHOLD"],
        "codeasdoc/mathematical_conventions.rst": ["LR-POLYNOMIAL", "LOG-MULTI-FOCUS"],
        "codeasdoc/animation_performance.rst": ["MOTION-FRAME-STEP", "MOTION-NATIVE-SLOW"],
        "codeasdoc/visual_design.rst": ["STYLE-THEME-ACADEMIC", "STYLE-FORMAT-DASHBOARD"],
        "codeasdoc/themes_formats_sizes.rst": ["STYLE-THEME-ACCESSIBLE", "STYLE-SIZE-CLASSROOM"],
        "codeasdoc/prediction_explanations.rst": [
            "LR-PRED-EXTRAP",
            "LR-PRED-10D-STAGES",
            "LOG-PRED-BINARY",
        ],
        "codeasdoc/export.rst": ["STYLE-RESPONSIVE", "NN-REPORT"],
        "codeasdoc/gallery.rst": [
            "SMOKE-LR-2D",
            "SMOKE-LOG-MULTI-2D",
            "NN-GRAPH-EXACT",
            "NN-SYNTH-REG-PREDICTION",
            "NN-REAL-IRIS-TRAINING",
            "NN-REAL-DIGITS-ARCHITECTURE",
        ],
        "codeasdoc/compatibility.rst": [
            "LR-PIPE-SCALED",
            "NN-RELU-ADAM",
            "NN-BLOCK-ATTENTION",
            "NN-BLOCK-MULTI-IO",
            "NN-REAL-CANCER-BLOCKS",
            "NN-REAL-DIGITS-TRAINING",
        ],
        "codeasdoc/model_hyperparameters.rst": [
            "HYPER-LR-L2",
            "HYPER-LOG-C-STRONG",
            "HYPER-NN-REGRESSION",
            "NN-BLOCK-CONV",
            "NN-BLOCK-LSTM",
            "NN-RECORDER-V2",
            "NN-ROUTER-HYPERPARAMETERS",
        ],
        "codeasdoc/limitations.rst": [
            "LR-MANY-FEATURES",
            "LOG-MULTI-MANY-CLASSES",
            "NN-BLOCK-DYNAMIC",
            "NN-BLOCK-COLLAPSE",
        ],
        "codeasdoc/visualization.rst": ["SMOKE-LOG-BIN-2D"],
        "codeasdoc/advanced.rst": ["LR-PRED-COUNTERFACTUAL", "LOG-MULTI-OVR"],
        "codeasdoc/architecture.rst": ["SMOKE-LR-ND", "NN-ARCHITECTURE"],
        "codeasdoc/contributing_visual_qa.rst": ["DATA-LR-SCALE", "DATA-LOG-OVERLAP"],
    }
    payload = {
        "schema_version": 1,
        "policy": "Every new or materially changed public documentation page adds a new visual QA cell.",
        "cases": cases,
        "documents": documents,
    }
    (NOTEBOOKS / "visual_case_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Generate every canonical notebook and its coverage manifest."""
    build_smoke()
    build_linear()
    build_polynomial()
    build_logistic()
    build_neural()
    build_neural_structures()
    build_motion()
    build_visual_system()
    build_hyperparameters()
    build_data_edges()
    build_learning()
    build_realistic_scenarios()
    build_manifest()


if __name__ == "__main__":
    main()
