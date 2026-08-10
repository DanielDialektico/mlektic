# Phase 1 — Mathematical parity for linear and logistic regression

## Objective

Bring tabular models to the same academic standard as the neural family while preserving visual approachability. A figure should teach the exact model mapping, parameter convention, prediction calculation, objective, and relevant optimization or interpolation semantics—not merely animate a line or surface.

## Shared pedagogical architecture

Every tabular model receives a common progressive structure:

1. **Definition:** symbols, feature vector, target, classes, and parameter dimensions.
2. **Model:** exact scalar and vector form.
3. **State:** current coefficients, intercept, feature space, and regularization settings.
4. **Substitution:** one concrete observation substituted into the model.
5. **Transformation:** identity for linear regression, sigmoid for binary logistic, Softmax or OvR normalization for multiclass.
6. **Decision:** numerical prediction, probability, threshold, or argmax.
7. **Objective:** empirical data term, regularization term, and normalization convention.
8. **Temporal context:** recorded/replayed/interpolated source and N/K.

The default density should remain readable. The complete chain belongs in an optional academic/detail mode, reusable report panel, or staged explanation rather than an overloaded default chart.

## Linear regression

### Required definitions

For one feature:

\[
\hat y = \theta_0 + \theta_1 x_1.
\]

For d features:

\[
\hat y_i = \theta_0 + \sum_{j=1}^{d}\theta_j x_{ij}
= \theta_0 + \mathbf{x}_i^\top\boldsymbol{\theta}.
\]

The figure must state the dimensions of \(\mathbf{x}\) and \(\boldsymbol{\theta}\), identify the intercept separately, and declare original vs scaled feature space.

### Objective and metrics

Show the exact empirical convention used by the library:

\[
\operatorname{MSE}(\boldsymbol{\theta},\theta_0)
= \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2.
\]

If the fitted estimator optimizes a differently normalized objective, document that distinction instead of claiming that displayed MSE is the optimizer's exact internal objective. R² and MAE are evaluation metrics; neither should be labeled as the training loss unless the estimator explicitly uses it.

### Regularization

When estimator parameters reveal regularization, show a separate term:

- L2: \(\lambda\lVert\boldsymbol{\theta}\rVert_2^2\)
- L1: \(\lambda\lVert\boldsymbol{\theta}\rVert_1\)
- elastic net: the estimator-specific weighted combination.

The intercept penalty convention must be stated. Unsupported or estimator-private objective details must be labeled “not introspected,” never inferred as exact.

### Gradient and update

For a mathematically valid replay, an academic view may show:

\[
\nabla_{\boldsymbol{\theta}}J
= -\frac{2}{n}\mathbf{X}^\top(\mathbf{y}-\hat{\mathbf{y}}),
\qquad
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}-\eta_t\nabla J.
\]

Only display the update as the estimator's exact rule if learning-rate schedule, penalty, averaging, batch behavior, and preprocessing are accounted for. Otherwise present it as the canonical gradient-descent reference equation.

### High-dimensional view

The nD view should replace decorative coefficient text with a compact mathematical report:

- vector equation and dimensions;
- a coefficient contribution table \(\theta_jx_j\);
- intercept and summed prediction;
- controlled truncation with “showing p of d features”;
- optional sorting by absolute contribution;
- feature names when supplied;
- a clear difference between coefficient magnitude and observation-specific contribution.

## Binary logistic regression

### Mathematical chain

\[
z_i = \theta_0 + \mathbf{x}_i^\top\boldsymbol{\theta},
\qquad
p_i = \sigma(z_i)=\frac{1}{1+e^{-z_i}},
\qquad
\hat y_i =
\begin{cases}
c_1,&p_i\ge \tau\\
c_0,&p_i<\tau.
\end{cases}
\]

The positive-class identity, fitted class order, and threshold \(\tau\) must be visible. Class labels may be numeric, strings, or other Scikit-learn-compatible scalar values.

### Loss

\[
\operatorname{LogLoss}
=-\frac{1}{n}\sum_{i=1}^{n}
[y_i'\log p_i +(1-y_i')\log(1-p_i)],
\]

where \(y_i'=\mathbb{1}[y_i=c_1]\). The class mapping must accompany the equation so students do not assume the positive class is always the integer 1.

### Geometry

The 1D view connects score, sigmoid, probability, threshold, and decision boundary. The 2D view distinguishes the score plane \(z(\mathbf{x})\), the probability surface \(\sigma(z)\), and the boundary \(z=0\). A toggle or staged lesson should prevent all three from competing simultaneously.

## Multiclass logistic regression

### Scores and links

\[
z_k(\mathbf{x}) = b_k + \mathbf{x}^\top\mathbf{w}_k.
\]

For multinomial models:

\[
p_k(\mathbf{x})=\frac{e^{z_k}}{\sum_{r=1}^{K}e^{z_r}}.
\]

For OvR models, compute classwise sigmoids and use the estimator-compatible normalization. The figure must identify which link was resolved and include the fitted `classes_` ordering that maps columns to labels.

### Surfaces

For two features and many classes, avoid presenting every probability surface at full opacity. Proposed modes:

- selected-class probability surface;
- predicted-region map;
- pairwise boundary view;
- compact small multiples when K is small.

The chosen class and “showing 1 of K” state must remain visible.

## Pipelines and transformations

For affine scaling, coefficients can be converted to original units and both forms can be documented:

\[
\mathbf{x}_s=(\mathbf{x}-\boldsymbol{\mu})\oslash\mathbf{s},
\quad
\mathbf{w}_o=\mathbf{w}_s\oslash\mathbf{s},
\quad
b_o=b_s-\mathbf{w}_s^\top(\boldsymbol{\mu}\oslash\mathbf{s}).
\]

For non-affine preprocessing, do not pretend a raw-space coefficient vector exists. Display transformed-feature mathematics and identify the preprocessing step. Feature-name propagation should use `get_feature_names_out` when available.

## Proposed API additions

```python
visualize_lr(
    ...,
    detail="essential",       # essential | academic | complete
    show_objective="auto",
    show_regularization="auto",
    feature_names=None,
    sample_index=None,
)

visualize_logistic(
    ...,
    detail="essential",
    threshold=0.5,
    class_focus=None,
    show_objective="auto",
    feature_names=None,
    sample_index=None,
)
```

Defaults remain equivalent to current behavior. New values must be validated centrally and recorded in figure metadata.

## Academic tests

- Figure predictions equal estimator predictions within declared tolerance.
- Displayed probabilities equal `predict_proba` for supported estimator/link combinations.
- Original/scaled coefficient transformations preserve predictions.
- Objective values match independent NumPy/Scikit-learn calculations.
- threshold and argmax decisions match fitted class ordering.
- contribution sums reconstruct the displayed score or prediction.
- regularization labels reflect estimator parameters and never include unsupported claims.

## Acceptance criteria

Phase 1 is complete when a student can derive the displayed prediction from the visible equations and numbers; linear, binary, and multiclass views share terminology; estimator-specific and canonical mathematics are distinguished; pipelines are represented honestly; and default figures remain approachable.
