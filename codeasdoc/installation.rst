============
Installation
============
The executable installation case is ``LEARN-GETTING-STARTED`` in
``notebooks/learn/learn_00_getting_started.ipynb``.
Core installation
=================

Mlektic requires Python 3.9 or newer. Install a released build with:

.. code-block:: bash

   pip install mlektic

For a source checkout:

.. code-block:: bash

   python -m pip install -e .

The core installs NumPy, Scikit-learn, and Plotly. PyTorch remains optional so a
tabular lesson does not acquire a large unrelated dependency.

Optional environments
=====================

.. code-block:: bash

   python -m pip install -e ".[torch]"       # neural visualizations
   python -m pip install -e ".[notebooks]"   # execute project notebooks
   python -m pip install -e ".[docs]"        # build Sphinx documentation
   python -m pip install -e ".[dev,torch]"   # maintainer validation

Verify the active environment:

.. code-block:: python

   import mlektic
   print(mlektic.__version__)

Notebook display troubleshooting
================================

Run the kernel from the same environment where Mlektic and Plotly are installed.
If a notebook displays no widget, use ``display(fig)`` or ``fig.show()`` and
confirm that the frontend permits Plotly JavaScript. HTML export provides a
frontend-independent handoff; see :doc:`export`.

The executable installation case is ``LEARN-GETTING-STARTED`` in
``notebooks/learn/learn_00_getting_started.ipynb``.
