Themes, formats, and sizes
============================================================

The classic dashboard remains the compatibility default. Independent opt-in
axes control presentation without changing estimator mathematics:

.. list-table:: Visual options
   :header-rows: 1

   * - Axis
     - Supported values
     - Meaning
   * - ``theme``
     - classic, academic, classroom, compact, accessible, aurora, editorial,
       blueprint
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

``aurora``, ``editorial``, and ``blueprint`` are *skins*: themes that change
only how a figure is painted. The figure is laid out exactly as ``classic`` lays
it out, so every title, equation, control, legend, and panel keeps its position,
size, and font. Pass them like any other theme, for example
``visualize_lr(model, X, y, theme="aurora")``.

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

Effects only add marks or recolour numbers. They are skipped on any axis that
computes its own range, so they cannot change a range, and neural figures keep
their own colour encodings: a skin recolours only their background, text, and
controls. ``figure.layout.meta["mlektic_visual"]["skin"]`` reports which
effects ran and why any were skipped.

The backdrop is a layout image inside the figure, so it shows in every renderer.
With ``responsive=True`` the canvas size is unknown in Python, and the skin falls
back to a solid background.

Editing a skin
--------------

The skins live in ``mlektic/visualization/skins`` and are split so the
presentation and the figure builders change independently:

``definitions.py``
   Pure data: each skin's palette, backdrop, and effect switches. Changing a
   colour or turning an effect off is a one-value edit here.

``roles.py``
   The only contract with the builders. It names what each trace is (data,
   model, loss, class curve, equation) from the ``uid`` or ``name`` the builder
   gives it. Builders never import the skins.

``styling.py``, ``backdrop.py``, ``effects.py``
   How a skin is painted. They assign colours, symbols, and dashes only.

``tests/test_skins.py`` compares every skinned figure with its classic twin after
removing paint, so a change that moved any element fails there. Inspect
``STYLE-SKIN-AURORA-MOTION`` next to ``STYLE-SKIN-AURORA-CLASSIC-TWIN``, plus
``STYLE-SKIN-BLUEPRINT-3D`` and ``STYLE-SKIN-EDITORIAL-NEURAL``.
