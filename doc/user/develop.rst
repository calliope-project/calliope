=================
Development guide
=================

Contributions are very welcome! See our `contributors guide on GitHub <https://github.com/calliope-project/calliope/blob/master/CONTRIBUTING.md>`_ for information on how to contribute.

The code lives on GitHub at `calliope-project/calliope <https://github.com/calliope-project/calliope>`_. Development takes place in the ``master`` branch. Stable versions are tagged off of ``master`` with `semantic versioning <http://semver.org/>`_.

Tests are included and can be run with ``py.test`` from the project's root directory.

Also see the list of `open issues <https://github.com/calliope-project/calliope/issues>`_,  planned `milestones <https://github.com/calliope-project/calliope/milestones>`_ and `projects <https://github.com/calliope-project/calliope/projects>`_ for an overview of where development is heading, and `join us on Gitter <https://gitter.im/calliope-project/calliope>`_ to ask questions or discuss code.

--------------------------------
Installing a development version
--------------------------------

As when installing a stable version, using ``conda`` is recommended.

If you only want to track the latest commit, without having a local Calliope
repository, then just download the `base.yml <https://raw.githubusercontent.com/calliope-project/calliope/master/requirements/base.yml>`_ and `latest.yml <https://raw.githubusercontent.com/calliope-project/calliope/master/requirements/latest.yml>`_ requirements files and run (assuming both are saved into a directory called requirements)::

    $ conda env create -n calliope_latest --file=requirements/base.yml --file=requirements/latest.yml

This will create a conda environment called ``calliope_latest``.

To actively contribute to Calliope development, you'll instead want to clone the repository, giving you an editable copy. This will provide you with the master branch in a known location on your local device.

First, clone the repository::

   $ git clone https://github.com/calliope-project/calliope

Using Anaconda/conda, install all requirements, including the free and open source GLPK solver, into a new environment, e.g. ``calliope_dev``::

   $ conda env create -f ./calliope/requirements/base.yml -n calliope_dev
   $ source activate calliope_dev

On Windows::

   $ conda env create -f ./calliope/requirements/base.yml -n calliope_dev
   $ activate calliope_dev

Then install Calliope itself with pip::

   $ pip install -e ./calliope

---------------------------
Creating modular extensions
---------------------------

As of version 0.6.0, dynamic loading of custom constraint generator extensions has been removed due it not not being used by users of Calliope. The ability to dynamically load custom functions to adjust time resolution remains (see below).

Time functions and masks
------------------------

Custom functions that adjust time resolution can be loaded dynamically during model initialisation. By default, Calliope first checks whether the name of a function or time mask refers to a function from the :mod:`calliope.core.time.masks` or :mod:`calliope.core.time.funcs` module, and if not, attempts to load the function from an importable module:

.. code-block:: yaml

   time:
      masks:
          - {function: week, options: {day_func: 'extreme', tech: 'wind', how: 'min'}}
          - {function: my_custom_module.my_custom_mask, options: {...}}
      function: my_custom_module.my_custom_function
      function_options: {...}

---------
Profiling
---------

To profile a Calliope run with the built-in national-scale example model, then visualise the results with snakeviz:

.. code-block:: shell

   make profile  # will dump profile output in the current directory
   snakeviz calliope.profile  # launch snakeviz to visually examine profile


Use ``mprof plot`` to plot memory use.

Other options for visualising:

* Interactive visualisation with `KCachegrind <https://kcachegrind.github.io/>`_ (on macOS, use QCachegrind, installed e.g. with ``brew install qcachegrind``)

   .. code-block:: shell

      pyprof2calltree -i calliope.profile -o calliope.calltree
      kcachegrind calliope.calltree

* Generate a call graph from the call tree via graphviz

   .. code-block:: shell

      # brew install gprof2dot
      gprof2dot -f callgrind calliope.calltree | dot -Tsvg -o callgraph.svg

-------------------------
Checklist for new release
-------------------------

Pre-release
-----------

* Make sure all unit tests pass
* Build up-to-date Plotly plots for the documentation with (``make doc-plots``)
* Re-run tutorial Jupyter notebooks, found in `doc/_static/notebooks`
* Make sure documentation builds without errors
* Make sure the release notes are up-to-date, especially that new features and backward incompatible changes are clearly marked

Create release
--------------

* Change ``_version.py`` version number
* Update changelog with final version number and release date
* Commit with message "Release vXXXX", then add a "vXXXX" tag, push both to GitHub
* Create a release through the GitHub web interface, using the same tag, titling it "Release vXXXX" (required for Zenodo to pull it in)
* Upload new release to PyPI: ``make all-dist``
* Update the conda-forge package:
    * Fork `conda-forge/calliope-feedstock <https://github.com/conda-forge/calliope-feedstock>`_, and update ``recipe/meta.yaml`` with:
        * Version number: ``{% set version = "XXXX" %}``
        * SHA256 of latest version from PyPI: ``{% set sha256 = "XXXX" %}``
        * Reset ``build: number: 0`` if it is not already at zero
        * If necessary, carry over any changed requirements from ``setup.py`` or ``requirements/base.yml``
    * Submit a pull request from an appropriately named branch in your fork (e.g. ``vXXXX``) to the `conda-forge/calliope-feedstock <https://github.com/conda-forge/calliope-feedstock>`_ repository

Post-release
------------

* Update changelog, adding a new vXXXX-dev heading, and update ``_version.py`` accordingly, in preparation for the next master commit

* Update the ``calliope_version`` setting in all example models to match the new version, but without the ``-dev`` string (so ``0.6.0-dev`` is ``0.6.0`` for the example models)

.. Note:: Adding '-dev' to the version string, such as ``__version__ = '0.1.0-dev'``, is required for the custom code in ``doc/conf.py`` to work when building in-development versions of the documentation.
