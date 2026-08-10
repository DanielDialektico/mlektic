# Phase 2 — Motion, sampling, and representation of time

## Objective

Preserve fluid visual learning while ensuring that fewer displayed states never imply fewer real or constructed states. Movement should communicate continuity; labels should communicate truth.

## Three-layer time model

### 1. Training or semantic history

This layer contains T training updates when genuinely recorded, or K semantic checkpoints when replayed/interpolated. Its values are model states, predictions, probabilities, losses, metrics, and parameters. These coordinates are never discarded from provenance metadata.

### 2. Display sampling

A sampling policy selects N checkpoints from K. It may be uniform, change-aware, loss-curvature-aware, or manually supplied. It must always retain required endpoints and expose the original coordinates.

### 3. Perceptual interpolation

Between retained checkpoints, the renderer may create q visual intervals. These subframes can interpolate trace coordinates, surface values, camera state, or other continuous visual attributes. Mathematical annotations and discrete metrics change at semantic checkpoint boundaries unless a quantity is explicitly and validly interpolated.

For a uniform hybrid path:

\[
F=(N-1)q+1.
\]

F is a rendering count, not an optimizer-step count.

## Visible timeline formula

The figure should expose the minimum relevant statement:

- recorded: `Showing N of K recorded checkpoints; T updates reported`;
- replayed: `Showing N of K reconstructed checkpoints; original fit was not recorded`;
- interpolated: `Showing N of K synthetic states; α = 0 → 1`;
- hybrid motion: optionally `q visual intervals between checkpoints` in metadata/help rather than the main title.

The slider label must show the source checkpoint, epoch, batch, or interpolation progress. Internal Plotly frame IDs are implementation details.

## Controls

### Slider

- semantic labels only;
- source coordinate retained after sampling;
- tooltip or help text with source and N/K;
- interpolation uses α or percentage;
- no duplicate slider labels after non-uniform sampling.

### Play/Pause

- maintain stable style during Plotly redraws;
- clearly separate playback speed from training semantics;
- allow replay from the current position;
- later add optional reduced-motion behavior without changing the animated default.

### Status line

An optional compact status line may show:

```text
Replay checkpoint 241/1000 · displayed 18/40 · EMA β=0.85 · original feature space
```

For interpolation:

```text
Synthetic interpolation α=0.62 · state 18/30 · no optimizer history
```

## Proposed configuration names

```python
visualize_lr(
    ...,
    history_steps=1000,
    display_checkpoints=40,
    sampling="uniform",
    visual_intervals=3,
    fps=36,
    transition="smooth",
)
```

Compatibility aliases remain for `steps`, `max_frames`, `frame_step`, and `interpolation_frames`. Documentation must define their layer explicitly before any future deprecation.

## Sampling strategies

### Uniform

Use evenly spaced source positions, retain endpoints, eliminate duplicate integer indices, and document the actual returned N when rounding reduces it.

### Cumulative change

Select checkpoints by cumulative distance in parameter or prediction space. This allocates more visible states where the model changes substantially and fewer where it is nearly stationary.

### Loss curvature

Retain endpoints, extrema, and high-curvature loss regions. This is useful for convergence lessons but should not be the default because it can bias students' perception of elapsed training time.

### Manual

Accept an explicit ordered unique coordinate sequence after validating range and endpoints. Manual selection is valuable for lesson authoring and regression snapshots.

## Fluidity by figure type

### 2D traces

Use trace-only interpolation when topology and point count remain stable. Preserve `line.simplify=False` where necessary. Mathematical layout updates should occur only at semantic frames to avoid expensive redraws.

### 3D surfaces

Plotly often requires redraws for surfaces. Optimize grid resolution, visible trace count, and semantic checkpoint count before removing motion. Consider selected-class views and perceptual subframes only after profiling browser memory and frame time.

### LaTeX

LaTeX strings do not interpolate meaningfully. Update them discretely at checkpoints. A continuous trace may move between two checkpoints while the formula panel states “values update at retained checkpoints.”

### Metrics

Do not interpolate discrete accuracy or class labels as if intermediate values were measured. Loss may be visually interpolated only as a display path; raw semantic values remain available.

## Performance budgets

Record, by representative notebook environment:

- payload generation time;
- figure construction time;
- HTML size;
- frame count and trace count;
- approximate browser memory;
- median and worst rendered frame time;
- Colab/Jupyter stability.

Proposed targets for ordinary laptop notebooks:

- first useful figure in under 3 seconds for small tabular examples;
- play interaction without multi-second stalls;
- default HTML below an agreed per-family budget;
- a documented high-detail mode when the default budget cannot preserve full geometry.

## Synchronization tests

- slider label matches the source coordinate used by coefficients, predictions, loss, and metrics;
- play/pause options retain configured frame and transition durations;
- hybrid subframes do not advance semantic labels early;
- first and final semantic checkpoints are always reachable;
- non-uniform sampling never mislabels a frame;
- raw and display timelines are both recoverable from metadata;
- trace arrays maintain stable length where interpolation requires it.

## Acceptance criteria

Phase 2 is complete when motion remains fluid, every visible state has an honest time coordinate, N/K is available, sampling and interpolation are separate concepts in API and documentation, 2D/3D/LaTeX behaviors are synchronized, and performance choices are evidence-backed rather than arbitrary.
