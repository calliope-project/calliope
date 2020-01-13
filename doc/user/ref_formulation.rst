------------------------
Mathematical formulation
------------------------

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

.. note::
    Make sure to also refer to the detailed :doc:`listing of constraints and costs along with their units and default values <config_defaults>`.

Decision variables
------------------

.. automodule:: calliope.backend.pyomo.variables
    :members:

Objective functions
-------------------

.. automodule:: calliope.backend.pyomo.objective
    :members:

.. _api_constraints:

Constraints
===========

Energy Balance
--------------

.. automodule:: calliope.backend.pyomo.constraints.energy_balance
    :members:

.. _constraint_capacity:

Capacity
--------

.. automodule:: calliope.backend.pyomo.constraints.capacity
    :members:

Dispatch
--------

.. automodule:: calliope.backend.pyomo.constraints.dispatch
    :members:

Costs
-----

.. automodule:: calliope.backend.pyomo.constraints.costs
    :members:

Export
------

.. automodule:: calliope.backend.pyomo.constraints.export
    :members:

MILP
----

.. automodule:: calliope.backend.pyomo.constraints.milp
    :members:

Conversion
----------

.. automodule:: calliope.backend.pyomo.constraints.conversion
    :members:

Conversion_plus
---------------

.. automodule:: calliope.backend.pyomo.constraints.conversion_plus
    :members:

Network
-------

.. automodule:: calliope.backend.pyomo.constraints.network
    :members:

Policy
------

.. automodule:: calliope.backend.pyomo.constraints.policy
    :members:

.. _constraint_group:

Group constraints
----------------------

.. automodule:: calliope.backend.pyomo.constraints.group
    :members:
