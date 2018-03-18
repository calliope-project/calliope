------------------------
Mathematical formulation
------------------------

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

Decision variables
------------------

**TBA - clean up and link to API**

Capacity
^^^^^^^^

* ``storage_cap(loc::tech)``: installed storage capacity. Supply plus/Storage only
* ``resource_cap(loc::tech)``: installed resource <-> storage/carrier_in conversion capacity
* ``energy_cap(loc::tech)``: installed resource/storage/carrier_in <-> carrier_out conversion capacity (gross)
* ``resource_area(loc::tech)``: resource collector area

Unit Commitment
^^^^^^^^^^^^^^^

* ``resource(loc::tech, timestep)``: resource <-> storage/carrier_in (+ production, - consumption)
* ``carrier_prod(loc::tech::carrier, timestep)``: resource/storage/carrier_in -> carrier_out (+ production)
* ``carrier_con(loc::tech::carrier, timestep)``: resource/storage/carrier_in <- carrier_out (- consumption)
* ``storage(loc::tech, timestep)``: total energy stored in technology
* ``carrier_export(loc::tech::carrier, timestep)``: carrier_out -> export

Costs
^^^^^

* ``cost(loc::tech, cost)``: total costs
* ``cost_investment(loc::tech, cost)``: investment operation costs
* ``cost_var(loc::tech, cost, timestep)``: variable operation costs

Binary/Integer variables
^^^^^^^^^^^^^^^^^^^^^^^^

* ``units(loc::tech)``: Number of integer installed technologies
* ``purchased(loc::tech)``: Binary switch indicating whether a technology has been installed
* ``operating_units(loc::tech, timestep)``: Binary switch indicating whether a technology that has been installed is operating

Objective function (cost minimization)
--------------------------------------

**TBA - link to API**

Provided by: :func:`calliope.constraints.objective.objective_cost_minimization`

The default objective function minimizes cost:

.. math::

   min: z = \sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{m}))

where :math:`cost_{m}` is the monetary cost class.

Alternative objective functions can be used by setting the ``objective`` in the model configuration (see :ref:`config_reference_model_wide`).

.. _api_constraints:

Constraints
===========

Energy Balance
--------------

.. automodule:: calliope.backend.pyomo.constraints.energy_balance
    :members:

Capacity
--------

.. automodule:: calliope.backend.pyomo.constraints.capacity
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
