
=================
Development guide
=================

The code lives on GitHub at `calliope-project/calliope <https://github.com/calliope-project/calliope>`_.

Development takes place in the ``master`` branch. Stable versions are tagged off of ``master`` with `semantic versioning <http://semver.org/>`_.

Tests are included and can be run with ``py.test`` from the project's root directory.

--------------------------------
Installing a development version
--------------------------------

With pip::

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

Post-release
------------

* Update changelog, adding a new vXXXX-dev heading, and update ``_version.py`` accordingly, in preparation for the next master commit

.. Note:: Adding '-dev' to the version string, such as ``__version__ = '0.1.0-dev'``, is required for the custom code in ``doc/conf.py`` to work when building in-development versions of the documentation.
