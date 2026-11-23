Themes, formats, and sizes
============================================================

Every figure is painted with the ``dusk`` skin unless another theme is
requested. The classic dashboard remains the geometric baseline: every theme
lays a figure out exactly like it, and ``theme="classic"`` gives the original
unskinned colours. Independent axes control presentation without changing
estimator mathematics:

.. list-table:: Visual options
   :header-rows: 1

   * - Axis
     - Supported values
     - Meaning
   * - ``theme``
     - classic, academic, classroom, compact, accessible, aurora, editorial,
       blueprint, dusk, chalkboard, davinci, ember, lumen
     - Color, typography, line, marker, and contrast tokens; skins repaint
       only
   * - ``format``
     - dashboard, compact, report
     - Composition and interaction structure
   * - ``density``
     - essential, academic, complete
     - Mathematical information depth
   * - ``size``
     - default, compact, notebook, wide, classroom
     - Named canvas dimensions

Explicit ``width`` and ``height`` override the selected size. ``responsive=True``
allows the resolved composition to scale with its container; it is not a
structural reflow engine. ``reduced_motion=True`` and ``format="report"`` show
the exact final displayed state without playback.

Inspect every preset independently in ``qa_05_visual_system.ipynb``, especially
``STYLE-THEME-ACCESSIBLE`` and ``STYLE-SIZE-CLASSROOM``.

Skins
=====

``aurora``, ``editorial``, ``blueprint``, ``dusk``, ``chalkboard``, ``davinci``,
``ember``, and ``lumen`` are *skins*: themes that change
only how a figure is painted. The figure is laid out exactly as ``classic`` lays
it out, so every title, equation, control, legend, and panel keeps its position,
size, and font. Pass them like any other theme, for example
``visualize_lr(model, X, y, theme="aurora")``.

``dusk`` is the default: omitting ``theme`` (or passing ``theme=None``) gives
exactly the same figure as ``theme="dusk"``. The default is exposed as
``mlektic.visualization.design.DEFAULT_THEME``.

``aurora``
   Night-blue gradient with luminous lines. It also adds motion cues: a fading
   trail of earlier model states and a soft glow under the model line, the loss
   curve's remaining path dotted with a marker on the current step, and equation
   numbers coloured green when they rose and coral when they fell since their
   last change.

``editorial``
   Warm paper background with ink colours, suited to projection and print.

``blueprint``
   Drafting-table blue with a fine grid, open data markers, and a dotted loss.

``dusk``
   Aurora's motion cues in muted slate, dusty rose, and sand: the same trail,
   glow, loss progress, and equation colours, with less saturation.

``chalkboard``
   A framed school slate: green with chalk dust, half-erased smudges, a wooden
   border, white and pastel chalk lines, open markers, and a chalk-style
   handwritten typeface.

``davinci``
   Leonardo's notebook: aged laid paper with uneven tone, foxing spots, and
   darkened edges; sepia ink, red-chalk model lines, a dashed loss, and a
   handwritten typeface.

``ember``
   Volcanic dark with fire colours, a strong glow, a long trail, and residual
   sparks: a segment from every data point to the model, which visibly shrinks
   as training reduces the error.

``lumen``
   Classic's colours on a charcoal backdrop with a warm light from below, with
   ember's glow, trail, loss progress, residual segments, and equation colours.

Some skins set a typeface for titles, axes, legends, and buttons (Lora for
``editorial``, Architects Daughter for ``blueprint``, Gaegu for
``chalkboard``, Kalam for ``davinci``). Equations are always typeset by MathJax
in its own mathematical font. Fonts load from Google Fonts with the figure;
without a connection, the listed system fallbacks are used.

In every skin the figure's buttons keep their colours on hover; the stylesheet
is scoped to each figure, so different themes can share one notebook page.
Three-dimensional class surfaces and class-coloured points use the skin's
class palette.

Effects only add marks or recolour numbers. They are skipped on any axis that
computes its own range, so they cannot change a range, and they never run on
neural figures. Neural figures still follow the skin: their encodings, drawn
for a dark canvas, are translated to the skin background first (neutral greys
keep their place between background and text, and on light skins saturated
hues are darkened to a readable lightness without changing their meaning),
then chrome and text are recoloured as in any other figure. ``figure.layout.meta["mlektic_visual"]["skin"]`` reports which
effects ran and why any were skipped.

The backdrop is a layout image inside the figure, so it shows in every renderer.
With ``responsive=True`` the canvas size is unknown in Python, and the skin falls
back to a solid background.

Editing a skin
--------------

The skins live in ``mlektic/visualization/skins`` and are split so the
presentation and the figure builders change independently:

``definitions.py``
   Pure data: each skin's palette, backdrop (gradient or grid, plus optional
   grain, mottling, smudges, laid lines, foxing, and vignette), effect switches
   (glow, trail, loss progress, equation colours, residuals), button hover
   colours, an optional CSS frame, and an optional typeface. Changing a colour
   or turning an effect off is a one-value edit here.

``roles.py``
   The only contract with the builders. It names what each trace is (data,
   model, loss, class curve, equation) from the ``uid`` or ``name`` the builder
   gives it. Builders never import the skins.

``styling.py``, ``backdrop.py``, ``effects.py``
   How a skin is painted. They assign colours, symbols, dashes, and the font
   family only.

``mlektic/visualization/controls.py``
   The per-figure stylesheet sent with the figure's HTML: button colours
   (including hover), the skin's frame, and its web font.

``tests/test_skins.py`` compares every skinned figure with its classic twin after
removing paint, so a change that moved any element fails there. Inspect
``STYLE-SKIN-AURORA-MOTION`` next to ``STYLE-SKIN-AURORA-CLASSIC-TWIN``, plus
``STYLE-SKIN-BLUEPRINT-3D``, ``STYLE-SKIN-EDITORIAL-NEURAL``,
``STYLE-SKIN-EMBER-RESIDUALS``, ``STYLE-SKIN-LUMEN-RESIDUALS``,
``STYLE-SKIN-CHALKBOARD-CURVES``, ``STYLE-SKIN-DAVINCI-3D``,
``STYLE-SKIN-DAVINCI-EXPLAINER``, and ``STYLE-SKIN-DUSK-BINARY``.
