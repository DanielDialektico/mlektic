"""Generate Mlektic's deterministic learning and human visual-QA notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    """Create a Markdown cell."""
    return nbf.v4.new_markdown_cell(text.strip())


def code(source: str, *, case_id: str | None = None, tags: list[str] | None = None):
    """Create a cleared code cell with stable project metadata."""
    metadata = {}
    if case_id:
        metadata["mlektic_case_id"] = case_id
    if tags:
        metadata["tags"] = tags
    return nbf.v4.new_code_cell(source.strip(), metadata=metadata, execution_count=None, outputs=[])


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
            "c=linear_case(2)\nquery=(c.X.max(axis=0)+1.5).reshape(1,-1)\ndisplay(explain_lr_prediction(c.model,c.X,c.y,x_query=query,format='lesson',size='wide'))",
        ),
        (
            "LR-PRED-COUNTERFACTUAL",
            "explicit user-provided counterfactual separated from model output",
            "c=linear_case(1)\ndisplay(explain_lr_prediction(c.model,c.X,c.y,x_query=[[0.2]],yhat=8.0,prediction_source='provided'))",
        ),
    ]
    for item in cases:
        cells.extend(case(*item))
    notebook(
        "QA 01 — linear regression",
        "Linear estimators, feature spaces, mathematical density, and prediction edge cases.",
        cells,
        "qa/qa_01_linear.ipynb",
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
            "non-default binary threshold in equations and boundary",
            "c=binary_case(2)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,threshold=0.7,detail='academic'))",
        ),
        (
            "LOG-BINARY-IMBALANCED",
            "imbalanced data with empirical log-loss and metrics",
            "c=binary_case(2,imbalanced=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,show_loss=True,detail='complete'))",
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
            "c=multiclass_case(2,string_labels=True)\ndisplay(visualize_logistic(c.model,c.X,c.y,steps=10,max_frames=6,class_focus='group-1',show_class_labels=True,detail='academic'))",
        ),
        (
            "LOG-MULTI-OVR",
            "normalized one-vs-rest semantics from SGDClassifier",
            "c=multiclass_case(2)\nfrom sklearn.linear_model import SGDClassifier\nm=SGDClassifier(loss='log_loss',max_iter=1000,random_state=17).fit(c.X,c.y)\ndisplay(visualize_logistic(m,c.X,c.y,steps=12,max_frames=6,multiclass_link='ovr',show_loss=True))",
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
            "display(visualize_nn_training(history,max_frames=6,format='lesson'))",
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
            "forward-pass substitutions evolving with retained parameter snapshots",
            "display(explain_nn_prediction(model,X[1],history=history,max_frames=6,theme='academic'))",
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
    notebook(
        "QA 03 — neural networks",
        "PyTorch architecture, genuine recorded training, parameters, activations, graph semantics, and predictions.",
        cells,
        "qa/qa_03_neural.ipynb",
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
playback time, never whether the model trace exists. Re-executing a lesson cell
resets it to stage **1 Data**; click **2 Model** or **4 Complete** to reveal the
model before playing the semantic frames.
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
            "MOTION-LESSON",
            "staged lesson composition with complete-stage recovery",
            "display(visualize_lr(linear.model,linear.X,linear.y,steps=18,max_frames=7,format='lesson',frame_duration=240))",
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
    notebook(
        "QA 04 — motion and temporal semantics",
        "Playback controls, decimation, smoothing, lesson stages, and static accessibility modes.",
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
    for fmt in ["dashboard", "lesson", "compact", "report"]:
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
                "display(visualize_lr(model,X,y,steps=12,max_frames=7,format='lesson',theme='classroom'))",
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
                "display(visualize_lr(c.model,c.X,c.y,steps=14,max_frames=8,detail='complete',format='lesson'))",
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
                "display(visualize_logistic(b.model,b.X,b.y,steps=14,max_frames=8,threshold=0.65,detail='complete',show_loss=True,format='lesson'))",
            ),
            *case(
                "LEARN-LOGISTIC-PREDICTION",
                "winning class index and optional semantic label",
                "display(explain_logistic_prediction(b.model,b.X,b.y,x_query=b.X[[5]],show_class_labels=True))",
            ),
            markdown("## Multiclass probability"),
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
                "display(visualize_nn_training(history,max_frames=8,format='lesson'))",
            ),
            *case(
                "LEARN-NN-GRAPH",
                "forward activations and backpropagation quantities",
                "display(visualize_nn_graph(model,X[1],history,max_frames=8,frame_duration=260))",
            ),
            *case(
                "LEARN-NN-PREDICTION",
                "numerical forward pass through retained parameters",
                "display(explain_nn_prediction(model,X[1],history=history,max_frames=8))",
            ),
        ],
        "learn_03_neural_networks.ipynb",
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
        "codeasdoc/neural_lesson.rst": ["LEARN-NN-ARCHITECTURE", "LEARN-NN-TRAINING"],
        "codeasdoc/history_semantics.rst": ["MOTION-NATIVE-SLOW", "MOTION-HYBRID-FLUID"],
        "codeasdoc/mathematical_parity.rst": ["LR-PIPE-ORIGINAL", "LOG-BINARY-THRESHOLD"],
        "codeasdoc/mathematical_conventions.rst": ["LR-POLYNOMIAL", "LOG-MULTI-FOCUS"],
        "codeasdoc/animation_performance.rst": ["MOTION-FRAME-STEP", "MOTION-LESSON"],
        "codeasdoc/visual_design.rst": ["STYLE-THEME-ACADEMIC", "STYLE-FORMAT-LESSON"],
        "codeasdoc/themes_formats_sizes.rst": ["STYLE-THEME-ACCESSIBLE", "STYLE-SIZE-CLASSROOM"],
        "codeasdoc/prediction_explanations.rst": ["LR-PRED-EXTRAP", "LOG-PRED-BINARY"],
        "codeasdoc/export.rst": ["STYLE-RESPONSIVE", "NN-REPORT"],
        "codeasdoc/gallery.rst": ["SMOKE-LR-2D", "SMOKE-LOG-MULTI-2D", "NN-GRAPH-EXACT"],
        "codeasdoc/compatibility.rst": ["LR-PIPE-SCALED", "NN-RELU-ADAM"],
        "codeasdoc/model_hyperparameters.rst": ["HYPER-LR-L2", "HYPER-LOG-C-STRONG", "HYPER-NN-REGRESSION"],
        "codeasdoc/limitations.rst": ["LR-MANY-FEATURES", "LOG-MULTI-MANY-CLASSES"],
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
    build_logistic()
    build_neural()
    build_motion()
    build_visual_system()
    build_hyperparameters()
    build_data_edges()
    build_learning()
    build_manifest()


if __name__ == "__main__":
    main()
