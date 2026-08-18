"""Generate Mlektic's deterministic learning and human visual-QA notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


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
            "display(visualize_nn_training(reg_history,max_frames=8,theme='academic',format='lesson'))",
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
        "codeasdoc/animation_performance.rst": ["MOTION-FRAME-STEP", "MOTION-LESSON"],
        "codeasdoc/visual_design.rst": ["STYLE-THEME-ACADEMIC", "STYLE-FORMAT-LESSON"],
        "codeasdoc/themes_formats_sizes.rst": ["STYLE-THEME-ACCESSIBLE", "STYLE-SIZE-CLASSROOM"],
        "codeasdoc/prediction_explanations.rst": ["LR-PRED-EXTRAP", "LOG-PRED-BINARY"],
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
    build_logistic()
    build_neural()
    build_neural_structures()
    build_motion()
    build_visual_system()
    build_hyperparameters()
    build_data_edges()
    build_learning()
    build_manifest()


if __name__ == "__main__":
    main()
