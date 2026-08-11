Export and publishing
============================================================

``export_figure`` writes an interactive HTML representation:

.. code-block:: python

   path = export_figure(
       fig,
       "lesson.html",
       include_plotly="inline",
       include_mathjax="cdn",
       responsive=False,
       auto_play=False,
   )

Inlining Plotly makes the interaction runtime offline-capable. The default
MathJax CDN still needs network access to render equations; setting
``include_mathjax=False`` preserves LaTeX source but cannot promise rendered
mathematics. Fully self-contained MathJax is not currently guaranteed.

For static publication use ``format="report"`` or ``reduced_motion=True`` to
select the exact final displayed state, then use Plotly's supported image export
tooling in the target environment. Responsive behavior is inherited from
``layout.meta['mlektic_visual']`` unless explicitly overridden.

Inspect ``STYLE-RESPONSIVE`` and ``NN-REPORT`` before publishing a new layout.
