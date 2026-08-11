# Phase 3 — Visual system, typography, spacing, and formats

> **Status:** Implemented on `feature/phase-3-visual-design`. The verified
> implementation decisions, compatibility guarantees, and test evidence are
> recorded in [the Phase 3 implementation record](10_phase_3_implementation_record.md).

## Objective

Make figures more elegant, compact, academically composed, and adaptable without changing the classic default. The neural family supplies the internal reference for hierarchy and restraint; tabular geometry remains the reference for spatial requirements.

## Comparative diagnosis

### Current tabular family

- strong color contrast and recognizable model/data separation;
- fixed 1100 × 600 layout and `autosize=False`;
- 24 px titles, large annotation regions, and repeated generous margins;
- formulas, controls, legend, metric cards, and geometry can compete vertically;
- 3D views require space, while nD report views often waste it;
- style values are centralized only partially.

### Current neural family

- more disciplined type scale and panel hierarchy;
- compact mathematical report composition;
- clearer separation of headings, equations, values, and notes;
- better use of cards and progressive views;
- visual vocabulary feels more like a polished academic tool.

## Design decision

Create an additive token and preset system. `classic` remains byte-for-byte equivalent where feasible. New presets opt into reduced typography, tighter spacing, responsive behavior, or accessibility enhancements.

The system has three independent axes:

1. **theme:** color, fonts, line weights, surfaces, controls;
2. **format:** figure composition and panel arrangement;
3. **density:** how much mathematical and contextual information is visible.

This prevents a “compact theme” from accidentally changing semantics or a “lesson format” from changing colors.

## Token model

```python
VisualTokens(
    font_family="Inter, Arial, sans-serif",
    math_font_family="STIX Two Math, serif",
    title_size=20,
    subtitle_size=12,
    section_size=15,
    body_size=13,
    equation_size=15,
    metric_size=13,
    control_size=12,
    space_xs=4,
    space_sm=8,
    space_md=12,
    space_lg=20,
    radius=8,
    line_width=2.5,
)
```

Plotly and MathJax font availability must be tested in Jupyter, Colab, exported HTML, and documentation screenshots. Web fonts should not become an undeclared network dependency.

## Proposed presets

### `classic` — default

- current dark palette;
- current 1100 × 600 size and fixed layout;
- current line widths and animation behavior;
- current typography unless a correctness fix requires a label change.

### `academic`

- more restrained title and annotation scale;
- compact subtitle for provenance and timeline;
- report-like equation panels;
- reduced top margin and deliberate whitespace;
- subtle panel borders rather than large empty zones.

### `classroom`

- larger text and controls for projection;
- simplified detail density;
- stronger line and marker separation;
- generous interaction targets.

### `compact`

- smaller height and margins;
- condensed legend and metrics;
- suitable for side-by-side notebook comparison;
- equations may move to staged controls or a report below the plot.

### `accessible`

- colorblind-safe palette;
- redundant line styles and markers;
- higher minimum contrast;
- no meaning conveyed by color alone;
- explicit focus and reduced-motion hooks.

## Proposed formats

### `dashboard` — compatibility default

Current combined geometry, formulas, controls, metrics, and optional loss layout.

### `lesson`

One concept at a time:

1. data;
2. model geometry;
3. parameter state;
4. substitution;
5. objective;
6. temporal interpretation.

The learner advances through semantic stages while motion remains available inside relevant stages.

### `compact`

Prioritizes geometry and one key equation. Details move into hover, metadata, or a companion report.

### `report`

Static or lightly interactive academic composition for papers, assignments, and exported documentation. It emphasizes exact definitions, estimator settings, and final-state calculations over playback controls.

## Compatibility API

```python
visualize_lr(
    ...,
    theme="classic",
    format="dashboard",
    density="essential",
    size="default",
    width=None,
    height=None,
    responsive=False,
)
```

Rules:

- omitting every new argument reproduces the current classic figure;
- explicit width/height overrides a named size preset;
- `responsive=True` is opt-in until cross-environment visual tests are stable;
- unknown themes, formats, density levels, and sizes fail clearly;
- figure metadata records resolved tokens and format.

## Responsive behavior

Responsive design has two levels:

### Scaling

The same composition scales inside a container. This works for moderate width changes but cannot solve crowded equations or legends.

### Reflow

Panels change arrangement at defined breakpoints. Proposed targets:

- wide desktop: geometry and report side by side;
- standard notebook: dashboard composition with compact header;
- narrow notebook: geometry above formulas/metrics;
- mobile/static: report or lesson format rather than an unusable compressed dashboard.

Because Plotly does not provide CSS-style subplot reflow automatically, formats may need separate layout builders sharing the same semantic view model.

## View-specific optimization

### Linear/logistic 1D

- reduce title and top annotation height;
- keep formula and slider from competing;
- use a compact metric strip;
- preserve smooth trace motion;
- keep extrapolation/provenance as a subtle subtitle.

### 3D

- allocate more horizontal and vertical geometry space;
- move dense formulas to a side card or staged report;
- limit simultaneous surfaces;
- keep camera position stable across frames;
- ensure sliders are not clipped in embedded output.

### nD

- favor contribution tables and compact equation cards;
- avoid fixed 3D-oriented dimensions;
- offer scroll/truncation intentionally;
- display feature-name coverage.

### Neural

- preserve the current elegant direction;
- adopt shared tokens only where it improves cross-family consistency;
- avoid forcing tabular controls into architecture or graph views.

## Accessibility and semantics

- add marker/line-shape redundancy;
- provide textual source, current checkpoint, and selected class;
- test contrast for text, controls, data, model, and boundaries;
- keep keyboard-reachable Plotly controls where possible;
- provide a static/report alternative for motion-sensitive users;
- ensure no critical explanation exists only in hover.

## Visual tests

Build deterministic HTML and screenshot fixtures for:

- every family and dimensionality;
- classic plus each new preset;
- dashboard, lesson, compact, and report formats;
- loss shown/hidden;
- long titles, string class labels, many features, many classes;
- original/scaled spaces;
- replay/interpolation subtitles;
- notebook widths at approximately 700, 1000, and 1400 px.

Measure overlaps, clipped controls, text truncation, axis readability, legend collisions, and title height. Classic snapshots form the compatibility baseline.

## Acceptance criteria

Phase 3 is complete when classic remains stable; academic and compact presets materially improve information density; formats solve different pedagogical contexts; explicit sizes and responsive behavior work in notebooks and exports; accessibility does not rely on color alone; and every supported combination has a visual regression fixture.
