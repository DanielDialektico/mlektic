# Phase 4 — QA notebooks and student learning notebooks

## Objective

Separate exhaustive library validation from pedagogical learning. The current notebooks are valuable exploratory assets, but a student should not need to navigate large debugging notebooks, and maintainers should not rely on visual memory to detect regressions.

## Audit of the four current notebooks

### `test_interpt.ipynb`

Role: broad experimentation with interpolation, animation behavior, and representative outputs.

Strengths:

- captures important edge cases and visual comparisons;
- useful for diagnosing frame construction and HTML behavior;
- contains examples that can seed formal regression cases.

Limitations:

- mixes exploratory cells, generated artifacts, and expected behavior;
- no machine-readable matrix of cases;
- large stored outputs obscure diffs;
- interpolation semantics are not consistently labeled in older outputs.

Proposal: preserve as an archive, extract minimal reproducible cases into `qa_00_smoke_matrix.ipynb`, and move deep motion experiments into `qa_04_motion.ipynb` if needed.

### `test_linreg.ipynb`

Role: linear variants across dimensions, pipelines, history modes, and visual options.

Strengths:

- broad estimator and dimensional coverage;
- exposes 1D, 2D, and nD layout behavior;
- tests prediction explanations and mathematical substitutions.

Limitations:

- cases are not indexed against supported configuration dimensions;
- assertions are mostly visual;
- repeated setup creates drift;
- large outputs make execution expensive.

Proposal: extract formal cases into `qa_01_linear.ipynb`; turn the clearest model derivation into `learn_01_linear_regression.ipynb`.

### `test_logreg.ipynb`

Role: binary/multiclass, 1D/2D/nD, pipeline, and probability visualization coverage.

Strengths:

- widest configuration surface;
- valuable coverage of multiclass links and 3D surfaces;
- reveals class-label, performance, and density issues.

Limitations:

- very large file and stored output footprint;
- difficult to know which cells are canonical;
- visual comparison is not automated;
- student narrative is secondary to exhaustive testing.

Proposal: split binary and multiclass sections inside `qa_02_logistic.ipynb`, generate cases from a table, and build `learn_02_logistic_regression.ipynb` around one binary and one multiclass derivation.

### `test_ann.ipynb`

Role: neural architecture, graph, training, weight, report, and prediction views.

Strengths:

- closest to a coherent public demonstration;
- validates the strongest design language in the project;
- useful as a reference for tabular mathematical rigor.

Limitations:

- still combines QA and showcase goals;
- environment and optional Torch requirements need clearer gates;
- outputs should be made deterministic where possible.

Proposal: create `qa_03_neural.ipynb` and a narrative `learn_03_neural_networks.ipynb`; preserve selected current figures as design references.

## Proposed repository structure

```text
notebooks/
  qa/
    qa_00_smoke_matrix.ipynb
    qa_01_linear.ipynb
    qa_02_logistic.ipynb
    qa_03_neural.ipynb
    qa_04_motion.ipynb
  learn/
    learn_00_getting_started.ipynb
    learn_01_linear_regression.ipynb
    learn_02_logistic_regression.ipynb
    learn_03_neural_networks.ipynb
  archive/
    ...current exploratory notebooks or links...
```

Do not move or delete the current notebooks until replacements reproduce their useful coverage.

## QA notebook design

### `qa_00_smoke_matrix.ipynb`

Purpose: fast execution of one representative case per major route.

Required cases:

- linear 1D, 2D, nD;
- binary logistic 1D, 2D, nD;
- multiclass logistic 1D, 2D, nD;
- replay and interpolation;
- pipeline and non-pipeline;
- original and scaled display space;
- one neural architecture/training smoke case when Torch is installed.

Each case records generation time, number of traces, semantic checkpoints, rendered frames, title/provenance text, and invariant assertions.

### `qa_01_linear.ipynb`

Generated matrix dimensions:

- estimator: `LinearRegression`, `SGDRegressor`, pipeline variants;
- dimensions: 1, 2, 5+;
- source: auto/replay/interpolation where valid;
- smoothing: none/EMA;
- loss: visible/hidden;
- animation: native/hybrid where supported;
- decimation: none/uniform/stride;
- query: in-range/extrapolation/counterfactual validation.

### `qa_02_logistic.ipynb`

Generated matrix dimensions:

- binary/multiclass;
- numeric/string labels;
- 1D/2D/nD;
- SGD/logistic regression/pipeline;
- Softmax/OvR/auto resolution;
- selected class and many-class truncation;
- probability and class verification failures;
- replay/interpolation, smoothing, and loss configurations.

### `qa_03_neural.ipynb`

Cover architecture, computational graph, training recorder, weight views, mathematical report, prediction explanations, optional convolutional models, and environments with/without optional dependencies.

## Student notebook design

### Shared structure

Every learning notebook follows:

1. learning objectives;
2. minimal environment setup;
3. small deterministic dataset;
4. model definition and fit;
5. prediction before visualization;
6. mathematical model definition;
7. Mlektic figure with provenance interpretation;
8. guided questions;
9. manual calculation for one observation;
10. comparison with estimator output;
11. experiment controls;
12. summary and limitations.

No notebook should hide model training, preprocessing, class order, or the source of history states.

### `learn_01_linear_regression.ipynb`

- line and hyperplane equations;
- coefficient units and scaling;
- MSE vs R²/MAE;
- contribution decomposition;
- replay vs interpolation;
- one extrapolation example;
- exercises changing slope, intercept, and feature scale.

### `learn_02_logistic_regression.ipynb`

- score, sigmoid, threshold, and log-loss;
- fitted class order;
- 2D decision boundary;
- Softmax vs OvR for a small multiclass example;
- calibrated interpretation caveat;
- exercises changing threshold and query position.

### `learn_03_neural_networks.ipynb`

- architecture and tensor dimensions;
- affine layer plus activation;
- forward pass for one sample;
- loss and optimizer distinction;
- recorded checkpoints vs displayed checkpoints;
- exercises changing hidden width, activation, and learning rate.

## Reusable helpers

Create test/notebook utilities for deterministic data, estimator factories, case IDs, timed rendering, invariant summaries, and artifact export. Helpers belong in a test-support module or documented examples package, not copied between notebooks.

## Output policy

- QA notebooks commit minimal or cleared outputs; CI executes them and stores artifacts separately.
- Learning notebooks may commit a small number of curated outputs so readers understand the result on GitHub.
- Large interactive HTML belongs in release/gallery artifacts, not repository root.
- Every generated artifact includes case ID, library version, environment, and source semantics.

## Automated execution

Use `nbclient`, `pytest`, or an equivalent checked workflow to:

- execute notebooks from a clean kernel;
- enforce per-cell and total timeouts;
- fail on exceptions and invariant violations;
- parameterize optional dependency cases;
- collect HTML/screenshot artifacts;
- compare lightweight semantic summaries in CI.

## Acceptance criteria

Phase 4 is complete when all useful current coverage is mapped; a smoke notebook runs quickly; QA cases are generated from explicit matrices; student notebooks form a coherent learning progression; outputs are controlled; notebooks execute from a clean environment; and stored artifacts no longer dominate repository size or obscure review.
