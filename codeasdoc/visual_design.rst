Visual design contract
======================

Phase 3 adds an optional design system without replacing Mlektic's classic
figures. The default remains the fixed-size dark dashboard with its existing
traces, controls, mathematical bands, frame counts, and playback cadence.

The design system is applied after model history, empirical evaluation, and
the mathematical contract have been constructed. Consequently, presentation
choices cannot change coefficients, probabilities, metrics, source
provenance, or temporal sampling.

Independent visual axes
-----------------------

Theme, format, and density solve different problems:

``theme``
   Controls typography, colors, line weights, markers, panels, and controls.

``format``
   Controls composition and interaction structure.

``density``
   Controls how much mathematical context is exposed. In tabular training
   views it is a compatible alias of :doc:`mathematical_parity`'s ``detail``.

Size and responsiveness are independent from those three axes.

.. code-block:: python

   from mlektic import visualize_lr

   figure = visualize_lr(
       model,
       X,
       y,
       theme="academic",
       format="dashboard",
       density="academic",
       size="notebook",
       responsive=True,
   )

Themes
------

``classic``
   Compatibility default. Omitting ``theme`` selects it.

``academic``
   Restrained type scale, subtle panels, and report-like contrast.

``classroom``
   Larger labels, controls, and lines for projected instruction.

``compact``
   Smaller type and spacing for notebook comparisons.

``accessible``
   Color-vision-safe colors plus non-color redundancy. Data markers are open,
   objective paths are dotted, decision boundaries are dashed, and multiclass
   marker symbols differ.

``aurora``, ``editorial``, ``blueprint``, ``dusk``, ``chalkboard``, ``davinci``, ``ember``, ``lumen``
   Skins: presentation-only themes that keep the classic geometry. See
   :doc:`themes_formats_sizes` for what each one paints and how to edit them.

The public immutable tokens can be inspected without constructing a figure:

.. code-block:: python

   from mlektic import available_themes, get_theme_tokens

   print(available_themes())
   print(get_theme_tokens("academic"))

Mlektic specifies local font-family fallbacks and does not download a web font.
Actual glyph metrics may vary slightly across notebook hosts.

Formats and motion
------------------

``dashboard``
   The established combined geometry, formula, metrics, objective, controls,
   and slider view.

``compact``
   Reduces canvas height and margins. It does not decimate frames or change
   mathematics.

``report``
   Applies the exact final displayed frame, removes playback controls, and
   reclaims the slider region. Use it for assignments, papers, documentation,
   and motion-sensitive contexts.

``reduced_motion=True`` is the explicit static alternative when another format
is otherwise desired. ``dashboard`` and ``compact`` preserve motion by
default. Animation speed remains controlled by ``frame_duration``,
``transition_duration``, ``fps``, and ``interpolation_frames``; Phase 3 does
not silently modify those values.

Density and Phase 1 detail
--------------------------

For ``visualize_lr`` and ``visualize_logistic``:

- ``density="essential"`` is the main mathematical animation;
- ``density="academic"`` adds the fitted-model derivation;
- ``density="complete"`` adds preprocessing, objective, regularization, and
  optimizer caveats where available.

If ``density`` is omitted, the existing ``detail`` argument remains
authoritative. Conflicting explicit non-default choices raise an error.

Sizes and responsive behavior
-----------------------------

Named sizes are ``default``, ``compact``, ``notebook``, ``wide``, and
``classroom``. Explicit pixel dimensions override them:

.. code-block:: python

   figure = visualize_logistic(
       classifier,
       X,
       y,
       theme="accessible",
       format="compact",
       width=900,
       height=650,
       responsive=True,
   )

Dimensions must be integer values of at least 320 pixels. Responsive mode sets
Plotly autosizing and removes a named preset width when no explicit width was
requested.

Scaling is not reflow
^^^^^^^^^^^^^^^^^^^^^

Responsive mode scales one resolved Plotly composition. Plotly does not
automatically rearrange subplots like a CSS grid. Select ``compact`` or
``report`` when the information structure itself must change for a narrow
notebook or static artifact. This limitation is recorded explicitly in figure
metadata.

Responsive export
-----------------

``export_figure`` inherits the visual contract by default:

.. code-block:: python

   responsive_figure = visualize_lr(model, X, y, responsive=True)
   export_figure(responsive_figure, "dashboard.html")

Pass ``responsive=True`` or ``False`` to ``export_figure`` to override it. A
figure without Phase 3 metadata remains fixed-size, preserving compatibility.

Titles, legends, and labels
---------------------------

Default titles describe the fitted model: the polynomial degree, other
non-affine preprocessing ("on Transformed Features"), the transformed
dimension, the regularized family (Ridge, Lasso, Elastic-Net), and :math:`K` for
multiclass models. A title longer than its canvas shrinks, to 14 px at least,
and clears the Play/Pause buttons.

Legends never hide data or labels. After sizing, every legend placed inside a
plot is compared with every point and line that plot draws in any frame, its
line shapes, and its annotations (metric cards, the threshold label). A legend
that covers any of them moves to the first free corner, then edge midpoint,
then free space inside the plot. Legends placed outside the plots and legends
of 3-D scenes are left alone. The decision is recorded in
``layout.meta["mlektic_visual"]["legend_placement"]`` as the position chosen
and the marks and labels still covered.

Hover labels use one solid background derived from the theme with the theme's
text colour, so they are readable over light and dark data alike. The
decision-threshold line uses the theme's text colour, and active control
buttons keep a readable label in every theme.

Audit metadata
--------------

Every public Phase 3 figure includes
``layout.meta["mlektic_visual"]``. It records:

- schema version and model family;
- resolved theme, format, density, and size;
- requested and resolved dimensions;
- all typography, spacing, color, line, and marker tokens;
- responsive and reduced-motion choices;
- whether animation frames remain;
- the responsive export configuration;
- accessibility and scaling-versus-reflow declarations;
- where each legend was placed (``legend_placement``).

This entry is merged with the existing history, mathematics, and prediction
contracts. It is intentionally JSON serializable.

Accessibility boundary
----------------------

The accessible theme avoids color-only distinctions for common scatter and
line views and provides a static alternative. It does not claim complete WCAG
conformance: Plotly, MathJax, Jupyter, Colab, browsers, and three-dimensional
surfaces contribute behavior outside Mlektic's control. Important model and
provenance information therefore remains available as visible text and
metadata rather than hover alone.

Prediction values and mathematical explanations use borderless annotations so
long numeric values cannot overflow a fixed box. Arrows and plotted markers
continue to identify the exact point and do not alter the reported value.

Public coverage
---------------

The shared visual contract applies to linear and logistic training figures,
their prediction explainers, and public neural architecture, graph, training,
weight, activation, and prediction views. Neural layouts preserve their
established elegant classic design; optional themes are applied without
forcing tabular subplot geometry onto them.
