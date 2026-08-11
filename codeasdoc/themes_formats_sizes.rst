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
     - classic, academic, classroom, compact, accessible
     - Color, typography, line, marker, and contrast tokens
   * - ``format``
     - dashboard, lesson, compact, report
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
