# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added `visualize_nn_hyperparameters()` and `view="hyperparameters"`, an
  instance-based PyTorch contract that lists every detected effective module,
  optimizer-group, objective, and scheduler setting without row truncation,
  pairs each value with its mathematical role and definition, and marks
  execution-only switches as non-mathematical. The neural recorder can now
  retain optional scheduler configuration for later lessons.
- Added separate neural fitted-prediction stages for Input, Substitution,
  Output, and Reset; ``parameter_state="training_replay"`` preserves the
  historical checkpoint animation as an explicit, distinct mode.
- Added an optional synchronized recorded-loss panel to dense neural graph
  animations, including the selected loss name and explicitly non-evaluated
  hollow markers for perceptual interpolation frames.
- Added ``visualize_nn_loss_landscape()`` for exact batch-loss evaluation on a
  disclosed affine two-direction parameter slice with a projected recorded
  optimization trajectory; it never labels the slice as the full landscape.
- Added ``visualize_nn_backpropagation()`` and ``view="backpropagation"`` with
  dense-layer chain-rule equations, globally comparable recorded layer
  gradient norms, adjacent parameter-update norms, and current loss, while
  distinguishing gradient computation from optimizer updates.
- Added ``math_font_scale`` to neural public figures as an additive way to
  enlarge LaTeX annotations whose size does not change with Plotly viewport
  zoom.
- Added a versioned renderer-independent neural execution graph, dual FX/eager
  capture with explicit provenance, semantic descriptors, and native Plotly
  block rendering for branches, merges, shared calls, multi-I/O, embeddings,
  recurrent networks, attention, Transformers, and transparent generic blocks.
- Added public `inspect_nn()`, `visualize_nn_blocks()`, opt-in
  `architecture_mode="blocks"` / `view="blocks"`, and
  `register_neural_descriptor()` extension APIs while preserving the legacy
  architecture view as the default.
- Added neural recorder history schema version 2 with independent buffers,
  effective optimizer groups, optional optimizer-state norms, temporal phase
  declarations, and buffer-aware historical replay.
- Expanded the neural QA notebook with a complete large-network matrix covering
  architecture, execution blocks, dense graph, prediction substitution,
  performance, weights, activations, backpropagation, and loss geometry, in
  addition to the existing public Plotly routes, synthetic and real datasets,
  and complex graph structures.

- Added the Phase-5 English documentation architecture: introduction,
  installation, rigorous linear/logistic/neural lessons, mathematical
  conventions, animation/performance, themes/formats/sizes, prediction
  explanations, export, compatibility, limitations, gallery, and contributing
  guides.
- Added twelve deterministic, output-cleared notebooks: eight maintainer human
  visual-QA matrices and four focused student lessons. Their 98 stable case IDs
  cover all public dimensional routes plus meaningful estimator, data,
  preprocessing, provenance, motion, mathematical-density, prediction, theme,
  format, size, accessibility, and neural-training variants.
- Added a machine-readable documentation-to-notebook manifest and validator.
  Every new or materially changed public documentation page must add a new real
  figure cell to its corresponding QA notebook; assertion-free cells, unique
  IDs, explicit display, cleared outputs, and complete mappings are CI enforced.
- Added deterministic notebook generation/execution tools, a local exploratory
  notebook archive with SHA-256 inventory, contributor guidance, Python
  3.9-3.13 test CI, optional PyTorch validation, documentation/package checks,
  clean wheel import smoke testing, PR notebook smoke execution, and scheduled
  full visual-notebook execution.
- Added complete package metadata, optional notebook/docs development extras,
  project URLs, classifiers, keywords, license metadata, and public
  ``mlektic.__version__``.
- Added a dedicated model/hyperparameter visual matrix covering linear
  ``alpha``, penalties and learning rates; logistic ``C``, penalties and class
  weights; and neural binary, multiclass, regression, deep-dense, and
  convolutional configurations.
- Added the opt-in Phase-3 visual system across linear, logistic, prediction,
  and neural public figures: immutable `VisualTokens`, `classic`, `academic`,
  `classroom`, `compact`, and `accessible` themes; `dashboard`, `lesson`,
  `compact`, and `report` formats; named sizes; explicit dimensions;
  responsive scaling; and reduced-motion final-state rendering.
- Added schema-versioned `layout.meta["mlektic_visual"]` metadata containing
  every resolved visual choice, token, dimension, motion decision,
  accessibility declaration, and responsive export setting.
- Added non-color redundancy in the accessible theme, staged pedagogical
  controls in lesson format, responsive HTML inheritance, a Phase-3 Sphinx
  guide, implementation record, and executable end-user showcase notebook.
- Added Phase-1 mathematical detail levels to `visualize_lr()` and `visualize_logistic()`: `detail="essential"` keeps the compact main visualization, while `"academic"` and `"complete"` add a stable fitted-model derivation without reducing animation frames or hybrid motion.
- Added estimator-backed `layout.meta["mlektic_math"]` contracts for linear, binary logistic, and multiclass logistic figures, including dimensions, feature space, parameters, one observation's contributions, reconstructed/model predictions, objective values, decisions, fitted class order, probability link, regularization settings, and canonical-versus-exact optimizer semantics.
- Added `show_objective`, `show_regularization`, `feature_names`, and `sample_index` controls to linear and logistic training figures; added binary `threshold` and multiclass `class_focus` controls to logistic figures.
- Added verified affine preprocessing conversion and transformed-feature mathematics for non-affine pipelines, using `get_feature_names_out` where available and explicitly declining to claim unavailable raw-space coefficients.
- Added a dedicated Sphinx mathematical-parity guide and an executable end-user Phase-1 notebook covering linear, binary, multiclass, pipeline, high-dimensional, and motion configurations.
- Added `show_class_labels=False` to logistic training and prediction figures. Indexed classes are the uncluttered default; fitted semantic labels can be revealed without removing their always-available metadata.
- Added `show_history_context=True` to linear and logistic public visualization APIs; setting it to `False` hides only the title subtitle while preserving slider and metadata context.
- Added a schema-versioned provenance contract for tabular histories, including replay/interpolation source detail, full and displayed checkpoint coordinates, estimator-reported iterations, smoothing/decimation settings, warnings, and final-state comparison.
- Added separate `loss_raw` and `loss_display` arrays while retaining `loss_hist` as a backward-compatible display alias.
- Added visible replay/interpolation subtitles and honest slider/loss-axis coordinates without changing classic geometry or animation defaults.
- Added strict validation for history configuration, training data, themes, prediction controls, supplied prediction/probability/class values, and unsupported explicit replay.
- Added extrapolation-aware linear and logistic prediction explanations, explicit counterfactual mode, string-label support, and auditable figure metadata.
- Added public `export_figure()` with explicit inline/CDN Plotly and MathJax dependency semantics.
- Added English-first history semantics, architecture, visualization, advanced-use, implementation-plan, and phase-0 verification documentation.
- Added hybrid 1D linear-regression animation with synchronized visual subframes for the model line, numeric coefficients, loss, and metrics; semantic checkpoints remain distinct in the slider and the symbolic definition stays in LaTeX.
- Added optional PyTorch support through `mlektic[torch]`, including `TorchTrainingRecorder` for frame-aligned loss, metrics, parameter values, gradients, compact activation vectors, and optimizer/loss metadata.
- Added a LaTeX-annotated architecture diagram with tensor dimensions, semantic layer shapes, formulas, configured hyperparameters, and compact summaries for large models.
- Added an animated mathematical network graph with one stable frame per training step, true global min/max scales for node outputs and edge weights by default, optional relative node contrast and forward-signal edge modes, simultaneous wine-red backpropagation overlays, final model tensors, and readable hover data without raw LaTeX syntax.
- Added a compact 2-by-2 learning-performance grid for loss and three independent metrics, automatic classification/regression metric inference from predictions and targets, and explicit empty metric panels when a history contains only loss.
- Added opt-in neural graph ``updates`` and ``hybrid`` evolution modes with
  signed parameter-update halos, previous-checkpoint or initial-state
  references, truthful global or explicit frame-normalized scales, top-k edge
  emphasis, update/gradient norm summaries, and smooth perceptual subframes
  that remain distinct from recorded optimizer checkpoints. The established
  absolute graph remains the default.
- Positioned neural update diagnostics in a dedicated band below the semantic
  timeline, with independent plotting coordinates so animated values remain
  legible without covering equations, nodes, edges, or color scales.
- Preserved neural activation heatmaps and mathematical color scales under
  additive themes, including ``accessible``; neural nodes are no longer
  restyled as generic data markers. The signal-color QA case now uses smooth
  perceptual subframes across every recorded checkpoint.
- Added a globally scaled forward-activity glow to every dense neural graph;
  signed color plus thickness and opacity encode
  ``s_ji = theta_ji * a_i`` independently of the opt-in parameter-update halo.
  The network band is vertically bounded so upper nodes cannot overlap its
  mathematical encoding legend.
- Added explicit dense-replay dropout disclosure: dropout remains visible in
  the semantic execution graph, while dense replay states that historical
  stochastic masks are not recorded and that activity glow is not a dropout
  encoding.

### Changed
- Replaced the obsolete README neural quick start with a complete recorder-v2
  workflow and current calls for architecture, graph replay, training,
  parameter evolution, fitted prediction, backpropagation, and effective
  hyperparameter mathematics. The guide now states capture timing, view
  separation, bounded-display semantics, and topology-routing guarantees.
- Separated neural training observation from fitted prediction completely:
  training replay now exposes only Play/Pause plus checkpoint navigation,
  parameter/signal equations, and no fixed-query/result cards or duplicated
  prediction; final prediction exposes only Input, numerical Substitution,
  Output, and Reset stages.
- Replaced the generic neural Substitution card with a fitted numerical
  substitution that multiplies actual query values by the corresponding fitted
  coefficients and reports the resulting first-unit pre-activation.
- Allocated neural forward equations by rendered line count and moved prediction
  controls to a reserved upper-left row, preventing multiline dense equations,
  activation formulas, and stage cards from overlapping.
- Replaced proportional neural forward-row spacing with an explicit MathJax
  baseline pitch and inter-layer corridor. Four-row Linear expansions now keep
  at least 0.08 paper coordinates before the following activation; replay uses
  its available upper band, and only genuinely dense layouts fall back from
  14-point to 13-point mathematics.
- Bounded any single expanded Linear derivation to four rendered rows; larger
  layers retain the exact computed vector but use the symbolic matrix form, so
  a deep lesson cannot consume the spacing reserved for subsequent layers.
- Increased the canonical large dense graph from five to eight sampled neurons
  per layer and made node diameter adapt to visible density, while retaining
  complete dimensions and sampled-count metadata.
- Split convolutional pedagogy explicitly between the complete execution-block
  topology and a bounded dense-classifier replay seeded by the actual first
  ``Linear`` input; the latter discloses omitted spatial modules in the figure
  and metadata.
- Preserved builder-reserved 12 px neural Play/Pause controls under every
  additive theme, including ``classroom``, instead of allowing theme typography
  to enlarge the buttons.
- Reduced the two overlaid neural loss-slice checkpoint labels from 17 px to
  15 px while preserving the surrounding title, axes, colorbar, and secondary
  explanatory typography.
- Increased spacing between graph activity/update legends, moved recorded
  objective headings safely inside their panels, and separated prediction
  controls from input and substitution headings.
- Made the per-edge recorded backpropagation overlay independently configurable
  through ``show_backpropagation`` and disabled it by default to reduce animated
  trace count. The neural QA gallery includes matched cases with and without it.
- Enlarged legacy neural architecture equations and hyperparameter labels,
  allocated parameter-evolution rows by rendered matrix height, increased the
  loss/update panel height, and labeled the final recorded loss directly on the
  three-dimensional loss slice without claiming convergence.
- Standardized neural graph header, colorbar, slider, and panel spacing;
  increased essential mathematical type; centered panel headings; and kept
  large-network readability through semantic/tensor truncation instead of
  shrinking all formulas.
- Redesigned the dedicated backpropagation animation with readable per-layer
  gradient, update, relative-update, loss, and loss-change values and slower
  default playback. Its lines remain redundant encodings, not the sole lesson.
- Reframed ``explain_nn_prediction()`` as an explicit Input / Substitution /
  Output view and added a winning-class result only when BCE or cross-entropy
  semantics justify it.
- Replaced the verbose middle summary in collapsed execution graphs with a
  compact visible operation count while retaining complete hover provenance.
- Moved learning-performance metric summaries farther above subplot titles and
  increased the upper margin.
- Rebalanced parameter-evolution typography and vertical spacing, replacing
  the disconnected floating vertical ellipsis with a plain-language count of
  omitted intermediate tensors.
- Moved neural parameter and forward-pass animation controls to a reserved upper-left area so buttons never cover the displayed equations.
- Fixed the mathematical-network parameter readout so weights and the training step update through animation traces without forcing a flickering redraw, and separated the title from the composed-function equation.
- Added explicit exact-zero and inactive-ReLU hover labels, inset the animated step readout, softened node outlines, and moved mathematical colorbar titles above their scales with wider outer margins.
- Added `explain_nn_prediction()` for time-aware layer-by-layer forward-pass mathematics, numerical substitutions for `z = Theta a + theta_0`, and summarized representations for deeper networks.
- Added standalone and notebook HTML mathematical reports with the complete taxonomy, definition, configuration, dimensions, parameter roles, and training evolution for every layer.
- Added reusable history metric builders for linear and logistic animations, including built-in support for `loss`, `mse`, `r2`, `mae`, `accuracy`, and `f1`, plus custom metric callables.
- Added reusable history sampling utilities to decimate long animation histories through `max_frames` or `frame_step`.
- Documented the adapter extension path for future model families, including non-Scikit-Learn estimators and upcoming neural-network visualization work.
- **2D Multiclass Logistic Regression Visualization**: Added full support for visualizing multi-class logistic regression in 2-dimensional feature spaces. The builder dynamically generates a 3D plot displaying the actual data points on the floor grid and $K$ distinct translucent, colored probability surfaces hovering and adjusting over time.
- Integrated a live LaTeX panel directly into the 2D multiclass layout, including the parameter matrix, bias vector, probability link, and step-by-step evaluation of representative class probabilities.
- Modified capture engines (`strategy_iterative.py` and `strategy_interp.py`) to systematically extract and cache a multidimensional probability surface history (`p_surfaces_hist`) required for $K$-class 3D rendering.
- Comprehensive `pytest` test suite covering `visualize_lr`, `visualize_logistic`, and `explain_lr_prediction` across 1D, 2D, and ND dimensional boundaries, including Pipeline scenarios.
- Moved legacy tests and old modules that were incompatible with Scikit-Learn to `old_mlektic_core`.
- Added an extreme complexity local test case (`local_test/log_test_2_var.py`) demonstrating `K=20` and `d=20` to validate robust mathematical rendering in maximum stress scenarios.
- `explain_lr_prediction()` and `explain_logistic_prediction()`: tools for mathematically and visually explaining predictions of already trained linear and logistic regression models.
- Dynamic metrics parameter (`metrics`) to display multiple variables (e.g., `loss`, `mse`, `r2`) simultaneously with smart number formatting to avoid visual overflows.
- Comprehensive initial documentation in `README.md`.
- Proper docstrings and enabled pydocstyle in ruff for Sphinx readiness.
- `frame_duration` control parameter to adjust animation speeds.
- Extensive local test cases matching notebook scenarios, including large multivariable tests (100 and 150 variables).

### Fixed
- Reserved a matrix-height row for bounded neural-weight omission notices, so
  the omitted-parameter count cannot cross either adjacent matrix in any
  animation frame. Crowded backpropagation readouts now alternate between two
  fixed rows, while omitted-layer scope disclosure occupies a separate lower
  caption below every gradient and update value.
- Replaced unbounded single-line configuration captions in legacy neural
  architecture figures with semantic multiline module columns. First and last
  captions anchor inward, classroom typography remains readable, and complete
  PyTorch configuration values remain available on hover.
- Rebuilt neural prediction staging around a fixed paper-coordinate section
  contract. The initial render is now Reset; Input, Substitution, and Output
  reveal cumulative fixed shapes whose borders are independent of asynchronous
  MathJax measurement. Summary vectors use at most two representative
  coordinates and three decimals, while the substitution preview uses a bounded
  three-row first-term / final-term / bias-and-result contract. Detailed vectors
  use a precision-aware width budget and aligned layer-owned formula blocks, so wide
  layers, deep networks, high decimal precision, themes, and public size
  presets cannot push equations outside their derivation column.
- Assigned neural graph content to invariant semantic rows for the model,
  parameter snapshot, training phase, optional backpropagation equation,
  heatmap legend, activity glow, update halo, and scope disclosure. Dynamic
  parameter/step traces use an invisible graph axis that spans every header
  row while node/edge content retains its bounded network domain. This keeps
  all graph variants aligned without placing the forward equation beside
  heatmap notation or clipping the evolving parameter snapshot when a loss
  panel is present.
- Reserved a 0.065 paper-coordinate clearance between the update-halo
  explanation and the uppermost node in every update-aware dense graph,
  including layouts with synchronized loss or diagnostics panels.
- Made neural loss-slice playback progressive: the initial state contains only
  its checkpoint marker, the projected path is revealed by frames, and the
  fixed 15 px final-loss annotation appears only at the recorded endpoint. The
  surface can no longer occlude or progressively shrink the final label, and a
  disclosed visual-only z-offset prevents WebGL from hiding the endpoint marker.
- Derived dense-graph node diameter from actual rendered vertical separation,
  leaving a real inter-node gap even when optional lower panels compress the
  graph. Numeric colorbar ticks now remain 11 px under every theme and size.
- Routed graph requests to the complete execution topology whenever dense
  replay would omit convolution, normalization, pooling, Dropout, tensor
  operations, shared calls, or branches. Execution formulas alternate above
  and below successive nodes, so CNN stages remain complete and their notation
  cannot collide.
- Prevented the generic lesson Data/Model/Objective/Complete menu from being
  added to neural performance figures; their four plots now remain visible and
  only the training Play/Pause controls and checkpoint slider are exposed.
- Made mathematical-architecture node dimensions density-aware for collapsed
  deep networks and replaced free-floating arrow glyphs with boundary-aware
  connectors. Arrow shafts and heads now occupy only the corridor between
  modules and never enter their semantic shapes.
- Increased the rendered gap between tall neural weight matrices and their
  matching bias vectors, and moved dropout/CNN scope disclosures below the
  activity-glow definition rather than alongside it.
- Fixed the multi-head-attention equation separator so MathJax receives
  ``\quad head_i`` instead of the invalid raw command ``\quadhead_i``.
- Expanded the mathematical safety padding of neural Substitution and Output
  cards to 18 px and lowered the first layer in short prediction diagrams so
  MathJax remains inside its borders without crowding the Input vector.
- Fixed static neural prediction reports and reduced-motion alternatives so
  removing staged controls first selects the complete Output explanation
  instead of leaving an input-only figure under a three-stage title.
- Contained the complete neural prediction output inside its bordered card and
  reserved a separate lower row for the highlighted final prediction so it no
  longer collides with the last activation substitution.
- Lifted multi-row neural weight matrices above their matching bias vectors
  using pair-aware rendered spacing, preventing MathJax matrix descenders from
  crowding the bias notation without moving the remaining parameter rows.
- Separated neural execution-block equations from node captions with a
  pixel-stable offset, and moved formulas to hover for dense parallel columns
  where an inline layout cannot remain collision-free.
- Replaced raw LaTeX in neural block hover cards with readable Unicode/plain
  mathematics, wrapped long attention/configuration details, and restored all
  observed output shapes for eager multi-output modules.
- Added theme-aware high-contrast boxes and borders to plotted prediction-value
  annotations (`y_hat` and `p_hat`) in linear, binary-logistic, multiclass, 2D,
  and 3D prediction explainers so values remain legible over model geometry.
- Prevented compact figures with academic/complete derivations from compressing
  the plot domain until the lower mathematics panel overlapped slider labels;
  the panel reserve and upper equation/title clearance now remain stable across
  named size presets and animated frames.
- Suppressed the expected Scikit-learn convergence warning produced only by Mlektic's intentional one-iteration replay initialization; warnings from the user's own estimator fit remain untouched.
- Synchronized coefficient-bearing logistic interpolation so every intermediate score, sigmoid/Softmax/OvR probability, curve or surface, and empirical loss is derived from the same interpolated parameter state; probability-only fallbacks are explicitly labeled.
- Restored normal 15-point typography and an inline coordinate tuple for ordinary two-feature linear prediction results; only genuinely long formatted coordinates now trigger a compact 13-point wrapped layout.
- Kept binary prediction axes numeric for string-labeled estimators so fitted sigmoid curves and probability surfaces remain visible; the two-feature boundary is now the actual `p=0.5` intersection line, and results identify both fitted-class probabilities and the winning label.
- Separated the multiclass 2D coefficient matrix and bias vector so the bias is centered between the matrix and class-score equation.
- Prevented normalized-OvR probability substitutions from crossing into loss panels by using exact compact fractions only in constrained replay layouts.
- Recentered multiclass ellipses in 1D and nD, removed empty rows from compact nD input vectors, and honored `max_theta_cols` in dense multiclass matrices.
- Removed empty loss subplots from linear and binary-logistic nD figures when `show_loss=False`.
- Shortened replay/interpolation provenance subtitles while preserving N/K, estimator iterations, and explicit endpoint-origin semantics.
- Balanced the multiclass probability stack by lowering only the first expanded fraction, keeping the ellipsis visually centered, and sharing the same fraction spacing and typography across 1D, 2D, and nD figures.
- Increased vertical separation between the definition and substituted sigmoid equations in binary logistic 2D figures so fraction numerators do not overlap the equation above.
- Fixed temporal decimation labels so retained checkpoints keep their source coordinates instead of being renumbered `0..N-1`.
- Fixed smoothed loss figures so the visible Loss/Log-loss metric uses the same display series while raw empirical values remain unchanged.
- Fixed 1D and 2D prediction ranges so out-of-range queries remain visible and are identified as extrapolations.
- Stabilized Play/Pause styling during 3D redraw animations by keeping buttons white with black text in every interaction state, without JavaScript state tracking or changes to interpolation.
- Moved hybrid 1D metric cards into a dedicated vertical side column so longer values cannot overlap each other or the loss plot.
- Prevented 2D prediction substitutions and results from overflowing by using compact LaTeX scientific notation, smaller panel typography, and a wrapped output tuple.
- Replaced display-only multicolumn grids for high-dimensional inputs and parameters with mathematically faithful truncated column vectors in linear and multiclass-logistic prediction explanations.
- Restored high-contrast metric cards in hybrid 1D animations using synchronized marker-and-text traces, preserving smooth playback without layout redraws.
- Restored animated LaTeX parameter and metric updates in Jupyter for layout-driven views; simple 1D regression now avoids layout redraws through synchronized numeric traces.
- Limited EMA smoothing to loss histories so displayed parameters always remain mathematically consistent with model geometry and probabilities.
- Prevented 2D regression lines from appearing partially drawn or blinking during playback by preserving every SVG line point and ensuring Plotly transitions finish before the next animation frame begins.
- Fixed multiclass logistic mathematics so figures and prediction explanations detect Softmax versus normalized one-vs-rest sigmoids instead of always claiming Softmax.
- Corrected ND parameter dimensions, matrix orientation, scalar output spaces, generic class notation, and the separation between `Theta` and `theta_0` across linear and logistic views.
- Replaced repeated expanded OvR denominators with compact exact normalizers to prevent LaTeX substitutions from overflowing into adjacent plots.
- Kept neural animation button labels readable when Plotly applies its light hover state.
- Fixed root-package exports so `from mlektic import explain_lr_prediction` matches the documented public API.
- Fixed logistic metric histories so classification metrics map predictions back to the original class labels instead of assuming zero-based label indexes.
- Preserved scaler metadata in interpolation histories so pipeline visualizations can evaluate metrics in the requested display space.
- **Plotly Visual Overflows**: Resolved massive text layout overlaps in logistic regression `prediction` views (1D, 2D, Multiclass 1D, and Multiclass ND) by precisely tuning Grid `column_widths`, `row_heights`, and expanding bounding boxes beyond default axis constraints.
- **Result Output Alignment**: Forced strict left-alignment on `\max` and `\hat{y} = \text{argmax}` math equations inside Multiclass result panels to prevent right-drifting.
- **Y-Axis Stand-Offs**: Prevented the $p(y=k \mid x)$ axis label from crashing into mathematical panels by directly shrinking `title_standoff` instead of disturbing layout proportions.
- **Plotly HTML Animation Rendering**: fixed an issue where exported HTML animations had unstable lines that cut or lengthened due to dynamic array resizing. Arrays (`loss_hist` and `step_axis`) are now padded with `None` to maintain a constant length across all frames.
- **Multivariable Prediction Formatting**: in `explain_lr_prediction` for $d \ge 3$, the output coordinate display string now properly appends $\hat{y}$ at the end (e.g., $(x_1, \ldots, x_d, \hat{y})$) to ensure mathematical consistency with 1D and 2D views.
- Fixed baseline value logic in `test_1_var.py`.

### Changed
- Reorganized notebooks by audience under ``notebooks/learn``, ``notebooks/qa``,
  and ``notebooks/archive``. Earlier phase notebooks retain Git history; seven
  very large exploratory notebooks remain preserved locally and ignored rather
  than being deleted or committed with stale embedded output.
- Preserved every retained animation frame in the new dashboard, compact, and
  lesson formats. Only the explicit report and reduced-motion alternatives
  freeze the exact final displayed state and remove temporal controls.
- Made `density` a validated alias of the Phase-1 `detail` contract for tabular
  training figures while leaving `detail` authoritative when density is
  omitted.
- Honored `show_loss=True` for synthetic interpolation across linear and logistic dimensionalities. These curves are labeled as interpolation MSE/log-loss rather than optimizer training loss; synthetic paths retain raw endpoint-exact values, replay-only EMA is declared explicitly, and compact linear models up to 10 variables use a shorter, better-balanced canvas with a wider loss panel.
- Wrapped fitted-model contribution expansions into dynamically spaced LaTeX rows. Moderate-dimensional panels show every coefficient-value product; higher-dimensional panels show a bounded, contribution-ranked selection and disclose the omitted count separately while preserving every value in metadata.
- Clarified synthetic interpolation with its baseline-to-fitted parameter equation and documented that polynomial-feature linear models can form nonlinear original-space geometry without implying gradient-descent training.
- Closed incremental linear and logistic replays with an explicitly labeled `fitted` endpoint from the supplied estimator. Intermediate states remain reconstructed, while the final equations, geometry, probabilities, raw loss, and evaluation metrics now match the model exactly; metadata records every state origin and the endpoint policy.
- Standardized one-dimensional linear-regression playback across all mathematical detail levels: the interpolated fitted equation now evolves in a dedicated LaTeX band above the plotting axes, metric cards retain their organized side column, and the equation no longer covers observations or the fitted line. The equation remains a trace update, so hybrid subframes keep `redraw=False`; academic panels remain separate below the slider.
- Advanced history metadata to schema version 2 with explicit coefficient-space semantics and parameter-versus-probability interpolation targets.
- Clarified that incremental Scikit-learn histories are reconstructed replays over clones and that non-incremental paths are synthetic interpolations, not recovered optimizer histories.
- Changed Sphinx documentation language to English and made the new public contract English-first.
- Changed `show_optimized()` to import IPython lazily so the core package import does not require notebook dependencies.
- Documented the animation rendering contract: hybrid 1D linear views keep symbolic LaTeX fixed and animate synchronized numeric traces, native views redraw dynamic MathJax substitutions, and notebook-friendly hybrid cadence is typically 30-45 FPS.
- Unified learnable-parameter notation across linear, logistic, and neural models: vectors use `theta`, matrices use `Theta`, and bias vectors use `theta_0`.
- Added configurable 2D animation transitions while preserving redraws for Plotly 3D traces; existing `steps`, `max_frames`, and `frame_step` controls remain available for temporal sampling.
- Completed the README, Sphinx guides, architecture reference, and public neural API docstrings for the PyTorch workflow, exact global heatmaps, optional relative/signal modes, inferred metrics, exact ReLU zeros, animated parameter readouts, and notebook/standalone HTML reports.
- Refactored `HistoryEngine` so it orchestrates capture, metric building, temporal sampling, smoothing, and parameter scaling through smaller focused helpers.
- Tightened Ruff configuration to lint the maintained package, tests, and Sphinx configuration while excluding generated, legacy, and manual-test directories.
- Updated README and Sphinx documentation for metrics, frame controls, `multiclass_2d`, generated API reference, and future adapter scalability.
- Replaced broad replay-configuration handling in the Scikit-Learn adapter with explicit parameter attempts and narrower exception handling.
- **Surface Smoothing (2D)**: Drastically enhanced rendering quality for Logistic Regression 2D probability surfaces and decision boundary planes by boosting the mesh grid resolution.
- **Layout Dimensions**: Enforced fixed `height` (and dimensions when appropriate) across visualization frames to guarantee consistent aspect ratios and eliminate bounding jitters.
- Refactored rendering core files (`prediction.py`, `multivar.py`, `binary_nd.py`, `multiclass_1d.py`) to eliminate Plotly trace code duplication, dynamically assemble subplot columns, and centralize LaTeX builders for better maintainability.
- Improved formatting and alignment of LaTeX mathematical annotations in the multivariable logistic regression visualization (`multiclass_nd.py`).
- Adjusted the layout of the logistic regression visualization for multi-class and multi-variable configurations to prevent overlapping of equations and matrices.
- Refined Multiclass Logistic prediction (1D and ND): replaced static matrix prediction display with dynamic mathematical argmax equations in the Result panel.
- Tightened mathematical substitution bounds and probability annotation bounds for high-cardinality multi-class classification (e.g. K=30) to eliminate vertical canvas overflow.
- Separated probability equations into aligned individual blocks (`\begin{aligned}`) allowing distinct configurations without compromising perfect equality alignment.
- Modified Y-axis positioning for numerical fraction and definitions to use available vertical canvas space properly.
- Re-synced formatting logic for the 1-graph vs 2-graphs layouts to make annotations mathematically identical (`multiclass_1d.py`).
- Refactored `last_class_tail_latex` to maintain proportional positioning relative to numerical and symbolic definitions.
- Automatically open HTML in browser for animations 1 and 2 in single variable tests.
- Set Plotly renderer to `notebook` by default in tests.
