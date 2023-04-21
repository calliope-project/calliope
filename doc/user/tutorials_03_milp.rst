.. _milp_example_model:

--------------------------------------------
Tutorial 3: Mixed Integer Linear Programming
--------------------------------------------
This example is based on the :ref:`urban scale example model <urban_scale_example>`, but with an override. In the model's ``scenarios.yaml`` file overrides are defined which trigger binary and integer decision variables, creating a MILP model, rather than a conventional LP model.

Units
=====

The capacity of a technology is usually a continuous decision variable, which can be within the range of 0 and ``flow_cap_max`` (the maximum capacity of a technology). In this model, we introduce a unit limit on the CHP instead:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/scenarios.yaml
   :language: yaml
   :dedent: 4
   :start-after: # chp-start
   :end-before: # chp-end

A unit maximum allows a discrete, integer number of CHP to be purchased, each having a capacity of ``flow_cap_per_unit``. ``flow_cap_max`` and ``flow_cap_min`` are now ignored, in favour of ``units_max`` or ``units_min``. A useful feature unlocked by introducing this is the ability to set a minimum operating capacity which is *only* enforced when the technology is operating. In the LP model, ``flow_out_min_relative`` would force the technology to operate at least at that proportion of its maximum capacity at each time step. In this model, the newly introduced ``flow_out_min_relative`` of 0.2 will ensure that the output of the CHP is 20% of its maximum capacity in any time step in which it has a non-zero output.

Purchase cost
=============

The boiler does not have a unit limit, it still utilises the continuous variable for its capacity. However, we have introduced a ``purchase`` cost:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/scenarios.yaml
   :language: yaml
   :dedent: 4
   :start-after: # boiler-start
   :end-before: # boiler-end

By introducing this, the boiler now has a binary decision variable associated with it, which is 1 if the boiler has a non-zero ``flow_cap`` (i.e. the optimisation results in investment in a boiler) and 0 if the capacity is 0. The purchase cost is applied to the binary result, providing a fixed cost on purchase of the technology, irrespective of the technology size. In physical terms, this may be associated with the cost of pipework, land purchase, etc. The purchase cost is also imposed on the CHP, which is applied to the number of integer CHP units in which the solver chooses to invest.

MILP functionality can be easily applied, but convergence is slower as a result of integer/binary variables. It is recommended to use a commercial solver (e.g. Gurobi, CPLEX) if you wish to utilise these variables outside this example model.

Asynchronous flow in/out
========================

The heat pipes which distribute thermal energy in the network may be prone to dissipating heat in an unphysical way. I.e. given that they have distribution losses associated with them, in any given timestep, a link could produce and consume energy in the same timestep, losing energy to the atmosphere in both instances, but having a net energy transmission of zero. This allows e.g. a CHP facility to overproduce heat to produce more cheap electricity, and have some way of dumping that heat. The ``async_flow_switch`` binary variable (triggered by the ``force_async_flow`` parameter) ensures this phenomenon is avoided:


.. literalinclude:: ../../src/calliope/example_models/urban_scale/scenarios.yaml
   :language: yaml
   :dedent: 4
   :start-after: # heat_pipes-start
   :end-before: # heat_pipes-end

Now, only one of ``flow_out`` and ``flow_in`` can be non-zero in a given timestep. This constraint can also be applied to storage technologies, to similarly control charge/discharge.

Running the model
=================

We now take you through running the model in a :nbviewer_docs:`Jupyter notebook, which you can view here <_static/notebooks/milp.ipynb>`. After clicking on that link, you can also  download and run the notebook yourself (you will need to have Calliope installed).
