
===========
Constraints
===========

-----------------
Basic constraints
-----------------

Objective function
------------------

Provided by: :func:`calliope.constraints.base.model_objective`

Node energy balance
-------------------

Provided by: :func:`calliope.constraints.base.node_energy_balance`

Node build constraints
----------------------

Provided by: :func:`calliope.constraints.base.node_constraints_build`

Node operational constraints
----------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_operational`

Transmission constraints
------------------------

Provided by: :func:`calliope.constraints.base.transmission_constraints`

Node costs
----------

Provided by: :func:`calliope.constraints.base.node_costs`

Model constraints
-----------------

Provided by: :func:`calliope.constraints.base.model_constraints`

--------------------
Optional constraints
--------------------

Ramping
-------

Provided by: :func:`calliope.constraints.ramping.ramping_rate`


----------------------------
Loading optional constraints
----------------------------

Additional constraints can be loaded by specifying two options in ``model.yaml``:

* ``constraints_pre_load:`` Will be evaluated just before loading constraints. Any Python code can be given here, for example ``import`` statements to import custom constraints.
* ``constraints:`` A list of constraints to load in addition to the default constraints, e.g. ``['constraints.ramping.ramping_rate']``

For example, the following settings would load two custom constraints from ``my_custom_module``::

   constraints_pre_load: 'import my_custom_module'
   contraints: ['my_custom_module.my_constraint0',
                'my_custom_module.my_constraint1']

Custom constraints have access to all model configuration (see :doc:`configuration`) and any number of additional configuration directives can be set on a per-technology, per-node or model-wide basis for custom constraints.
