# Test and acceptance matrix

## Purpose

Define the evidence required to call each phase complete. Unit tests protect mathematics and contracts; semantic figure tests protect synchronization; screenshots protect layout; notebooks protect real learning workflows.

## Tabular coverage dimensions

### Families and dimensionality

- linear: 1D, 2D, nD;
- logistic binary: 1D, 2D, nD;
- logistic multiclass: 1D, 2D, nD;
- prediction explainers for every supported dimensional route.

### Sources

- recorded where a real recorder exists;
- replayed with final match true/false/unknown;
- interpolated with each supported baseline;
- auto resolution;
- invalid explicit replay on non-incremental estimators.

### Estimators

- `LinearRegression`;
- `SGDRegressor`;
- `LogisticRegression`;
- `SGDClassifier`;
- direct estimator and `Pipeline`;
- affine scaler present/absent;
- estimator without extractable coefficients;
- numeric and string class labels.

### Data

- small/medium sample counts;
- constant and low-variance features;
- very different feature scales;
- imbalanced binary classes;
- three and many multiclass labels;
- constant targets where R² needs a defined convention;
- non-finite and mismatched invalid inputs;
- in-range and extrapolation queries;
- many features and long feature names.

## Neural coverage dimensions

- dense, convolutional, and supported custom module shapes;
- regression/binary/multiclass output semantics;
- architecture and computational graph views;
- recorded training with different `record_every` values;
- training loss and metrics;
- parameter/weight views;
- mathematical report and prediction explanation;
- CPU and optional accelerator behavior;
- Torch installed/absent.

## Mathematical invariants

### Linear

- \(\hat y = Xw+b\) matches the estimator in the declared space;
- original/scaled coefficient transformation preserves predictions;
- displayed MSE/MAE/R² match independent calculations;
- contribution sum plus intercept equals displayed prediction;
- interpolation α=1 equals the fitted model.

### Logistic

- binary score and sigmoid match estimator semantics;
- multiclass probabilities match resolved Softmax/OvR semantics;
- probability rows sum to one;
- class order matches `classes_`;
- threshold/argmax decisions match displayed classes;
- log-loss and F1/accuracy match independent calculations.

### Timeline

- source coordinates are monotonic and endpoints retained;
- K equals source-aligned array length before decimation;
- N equals every displayed time-aligned array length;
- slider labels correspond to retained coordinates;
- interpolation uses α/progress, not training-step terminology;
- hybrid visual frames do not masquerade as semantic states.

### Smoothing

- raw arrays do not mutate;
- no smoothing means raw and display values are equal but independently owned;
- EMA output matches the reference implementation;
- visible loss card and curve use the same display series;
- metadata states method and β.

## Minimum visual matrix

| Case | Classic baseline | Academic | Compact | Narrow width | Exported HTML |
|---|---:|---:|---:|---:|---:|
| Linear 1D replay | required | required | required | required | required |
| Linear 2D interpolation | required | required | optional | required | required |
| Linear nD | required | required | required | required | required |
| Binary logistic 1D | required | required | required | required | required |
| Binary logistic 2D | required | required | optional | required | required |
| Multiclass 1D | required | required | required | required | required |
| Multiclass 2D | required | required | optional | required | required |
| Logistic nD | required | required | required | required | required |
| Neural architecture | required | required/shared | optional | required | required |
| Neural training/report | required | required/shared | required | required | required |

## Visual checks

For every fixture verify:

- no overlap between title, subtitle, formulas, controls, legend, and slider;
- no clipped slider or buttons;
- axis labels and class identity are readable;
- geometry remains visible when query is outside training range;
- 3D camera and ranges are stable across frames;
- metric values correspond to the current semantic checkpoint;
- truncated content says how much is hidden;
- provenance and N/K are discoverable;
- classic snapshots remain within an approved tolerance.

## Performance matrix

Record for small, medium, and stress cases:

- history construction time;
- figure construction time;
- semantic checkpoints K;
- displayed checkpoints N;
- visual frames F;
- trace count;
- serialized HTML bytes;
- notebook display success;
- interaction/frame timing where automated browser tooling is available.

Performance regressions require an explicit tradeoff note. A faster result that removes pedagogically necessary motion or hides states does not pass.

## API compatibility

- current default calls retain classic colors, dimensions, and motion;
- new parameters are optional;
- compatibility aliases produce equivalent results and warnings only when documented;
- invalid enum values raise actionable English messages;
- public imports work without optional dependencies;
- metadata schema changes are versioned;
- HTML export dependency behavior is testable.

## Documentation matrix

Every public feature requires:

- API reference entry;
- accepted values and defaults;
- minimal executable example;
- mathematical definition where applicable;
- provenance/source explanation;
- limitation and performance note;
- expected visual or static fallback;
- notebook cross-link when pedagogically useful.

## Definition of Done

### Phase 0

- provenance metadata and visible source labels;
- preserved temporal indices;
- raw/display loss split;
- strict configuration and prediction validation;
- reliable documented HTML export;
- English contract documentation;
- unit tests, Ruff, and Sphinx pass.

### Phase 1

- shared mathematical grammar;
- exact/canonical estimator mathematics distinguished;
- objective, metrics, regularization, feature space, and class identity explicit;
- nD contributions and prediction calculations verified.

### Phase 2

- K/N/q/F semantics implemented;
- sampling and perceptual interpolation separated;
- synchronization and performance evidence;
- motion preserved as a default learning feature.

### Phase 3

- tokenized design system;
- classic unchanged;
- academic/compact/accessible alternatives;
- format and responsive tests;
- accessibility checks.

### Phase 4

- current notebook coverage mapped;
- generated QA matrix;
- focused student progression;
- clean-kernel automated execution;
- controlled artifacts.

### Phase 5

- clean package install;
- English documentation builds cleanly;
- gallery and notebooks reproduce;
- optional dependencies, compatibility, and limitations verified;
- release candidate evidence assembled.

## General release decision

Mlektic is ready to be presented as a public pedagogical tool when no animation can be reasonably misread about its source, every visible value has a mathematical or estimator-backed explanation, defaults remain stable, motion is fluid and semantically honest, and a student can follow the learning path without maintainer assistance.
