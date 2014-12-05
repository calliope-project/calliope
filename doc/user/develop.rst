
=================
Development guide
=================

The code lives on GitHub at `sjpfenninger/calliope <https://github.com/sjpfenninger/calliope>`_.

Development takes place in the ``master`` branch. Stable versions are tagged off of ``master`` with `semantic versioning <http://semver.org/>`_.

Tests are included and can be run with ``py.test`` from the project's root directory.

--------------------------------
Installing a development version
--------------------------------

With pip::

   pip install -e git+https://github.com/sjpfenninger/calliope.git#egg=calliope

Or, for a more easily modifiable local installation, first clone the repository to a location of your choosing, and then install via pip::

   git clone https://github.com/sjpfenninger/calliope
   pip install -e /path/to/your/cloned/repository

---------------------------
Creating custom constraints
---------------------------

By making use of the ability to load custom constraints (see :ref:`loading_optional_constraints`), a Calliope model can be extended by additional constraints easily without modifying the core code.

Constraint functions are called during construction of the model with the :class:`~calliope.Model` object passed as the only parameter.

The ``Model`` object provides, amongst other things:

* The Pyomo model instance, under the property ``m``
* The model data under the ``data`` property
* An easy way to access model configuration with the :meth:`~calliope.Model.get_option` method

A constraint function can add constraints, parameters, and variables directly to the Pyomo model instance (``Model.m``). Refer to the `Pyomo documentation <https://software.sandia.gov/trac/pyomo/>`_ for information on how to construct these model components.

The default cost-minimizing objective function provides a good example:

.. literalinclude:: ../../calliope/constraints/objective.py
   :language: python
   :lines: 12-

See the source code of the :func:`~calliope.constraints.ramping.ramping_rate` function for a more elaborate example.

