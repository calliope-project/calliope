
=================
Development guide
=================

The code lives on GitHub at `calliope-project/calliope <https://github.com/calliope-project/calliope>`_.

Development takes place in the ``master`` branch. Stable versions are tagged off of ``master`` with `semantic versioning <http://semver.org/>`_.

Tests are included and can be run with ``py.test`` from the project's root directory.

See the list of `open issues <https://github.com/calliope-project/calliope/issues>`_ and planned `milestones <https://github.com/calliope-project/calliope/milestones>`_ for an overview of where development is heading, and `join us on Gitter <https://gitter.im/calliope-project/calliope>`_ to ask questions or discuss code.

--------------------------------
Installing a development version
--------------------------------

First, with Anaconda installed, you can create a new Python 3.5 environment with all the supporting modules, including the free and open source GLPK solver. The easiest method is to download the latest `requirements.yml <https://github.com/calliope-project/calliope/blob/master/requirements.yml>`_ file, changing the environment name in the file from ``calliope``, if desired. Then create the environment::

    $ conda env create -f requirements.yml

Within your environment, install Calliope with pip::

   $ pip install -e git+https://github.com/calliope-project/calliope.git#egg=calliope

Or, for a more easily modifiable local installation, first clone the repository to a location of your choosing, and then install via pip::

   $ git clone https://github.com/calliope-project/calliope
   $ pip install -e ./calliope

---------------------------
Creating modular extensions
---------------------------

Constraint generator functions
------------------------------

By making use of the ability to load custom constraint generator functions (see :ref:`loading_optional_constraints`), a Calliope model can be extended by additional constraints easily without modifying the core code.

Constraint generator functions are called during construction of the model with the :class:`~calliope.Model` object passed as the only parameter.

The ``Model`` object provides, amongst other things:

* The Pyomo model instance, under the property ``m``
* The model data under the ``data`` property
* An easy way to access model configuration with the :meth:`~calliope.Model.get_option` method

A constraint generator function can add constraints, parameters, and variables directly to the Pyomo model instance (``Model.m``). Refer to the `Pyomo documentation <https://software.sandia.gov/trac/pyomo/>`_ for information on how to construct these model components.

The default cost-minimizing objective function provides a good example:

.. literalinclude:: ../../calliope/constraints/objective.py
   :language: python
   :lines: 12-

See the source code of the :func:`~calliope.constraints.optional.ramping_rate` function for a more elaborate example.

The process of including custom, optional constraints is as follows:

First, create the source code (see e.g. the above example for the ``ramping_rate`` function) in a file, for example ``my_constraints.py``

Then, assuming your custom constraint generator function is called ``my_first_custom_constraint`` and is defined in ``my_constraints.py``, you can tell Calliope to load it by adding it to the list of optional constraints in your model configuration as follows::

  constraints:
      - constraints.optional.ramping_rate
      - my_constraints.my_first_custom_constraint

This assumes that the file ``my_constraints.py`` is importable when the model is run. It must therefore either be in the directory from which the model is run, installed as a Python module (see `this document <https://python-packaging.readthedocs.io/en/latest/index.html>`_ on how to create importable and installable Python packages), or the Python import path has to be adjusted according to the `official Python documentation <https://docs.python.org/3/tutorial/modules.html#the-module-search-path>`_.

Subsets
-------

Calliope internally builds many subsets to better manage constraints, in particular, subsets of different groups of technologies. These subsets can be used in the definition of constraints and are used extensively in the definition of Calliope's built-in constraints. See the detailed definitions in :mod:`calliope.sets`, an overview of which is included here.

.. include:: ../../calliope/sets.py
   :start-after: ###PART TO INCLUDE IN DOCUMENTATION STARTS HERE###
   :end-before: ###PART TO INCLUDE IN DOCUMENTATION ENDS HERE###

Time functions and masks
------------------------

Like custom constraint generator functions, custom functions that adjust time resolution can be loaded dynamically during model initialization. By default, Calliope first checks whether the name of a function or time mask refers to a function from the :mod:`calliope.time_masks` or :mod:`calliope.time_functions` module, and if not, attempts to load the function from an importable module:

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

To profile a Calliope run with the built-in national-scale example model, then visualize the results with snakeviz:

.. code-block:: shell

   make profile  # will dump profile output in the current directory
   snakeviz calliope.profile  # launch snakeviz to visually examine profile


Use ``mprof plot`` to plot memory use.

Other options for visualizing:

* Interactive visualization with `KCachegrind <https://kcachegrind.github.io/>`_ (on macOS, use QCachegrind, installed e.g. with ``brew install qcachegrind``)

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
        * MD5 of latest version from PyPI: ``{% set md5 = "XXXX" %}``
        * Reset ``build: number: 0`` if it is not already at zero
        * If necessary, carry over any changed requirements from ``requirements.yml`` or ``setup.py``
    * Submit a pull request from an appropriately named branch in your fork (e.g. ``vXXXX``) to the `conda-forge/calliope-feedstock <https://github.com/conda-forge/calliope-feedstock>`_ repository

Post-release
------------

* Update changelog, adding a new vXXXX-dev heading, and update ``_version.py`` accordingly, in preparation for the next master commit

.. Note:: Adding '-dev' to the version string, such as ``__version__ = '0.1.0-dev'``, is required for the custom code in ``doc/conf.py`` to work when building in-development versions of the documentation.
