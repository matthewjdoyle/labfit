.. _tutorial:

Practical laboratory tutorial
=============================

Work through six synthetic experiments, from a first calibration fit to
overlapping spectral peaks and correlated thermal measurements. The tutorial
is written for undergraduate scientists with basic Python knowledge and
includes executable analyses, observations, figures and computed results.

The progression
---------------

* **Simple:** linear sensor calibration and exponential signal decay.
* **Intermediate:** a spectral peak on a sloping background and repeated
  damped-oscillation measurements.
* **Complex:** custom overlapping peaks and generalized least squares with
  correlated measurement errors.

Each example explains the model and uncertainty assumptions, interprets the
fitted parameters, and checks residuals. All observations are explicitly
synthetic; their equations, units, noise specifications and seeds are supplied.

Downloads
---------

* :download:`Printable PDF tutorial <../examples/tutorial/tutorial.pdf>`
* :download:`Markdown tutorial <../examples/tutorial/tutorial.md>`
* :download:`Editable LaTeX source <../examples/tutorial/tutorial.tex>`
* :download:`Complete tutorial bundle <../examples/tutorial/tutorial_bundle.zip>`

Download the complete bundle to keep the Markdown images and LaTeX vector
figures with their documents. It includes all scripts and CSV data.

Run from a checkout
-------------------

.. code-block:: shell

   python -m pip install -e .
   python examples/tutorial/run_examples.py

To regenerate synthetic data explicitly and rebuild both formats:

.. code-block:: shell

   python examples/tutorial/generate_data.py
   python examples/tutorial/build_tutorial.py

The document build requires Pandoc and Tectonic. The Markdown-only build uses
``--markdown-only`` and requires only the ordinary Python analysis dependencies.
See ``examples/tutorial/README.md`` for standalone example commands and build options.

