.. _milp_example_model:

--------------------------------------------
Tutorial 3: Mixed Integer Linear Programming
--------------------------------------------
This example is based on the :ref:`urban scale example model <urban_scale_example>`, but with an override. An override file exists in which binary and integer decision variables are triggered, creating a MILP model, rather than the conventional Calliope LP model.

.. Warning::

   Integer and Binary variables are still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behaviour.

Units
=====

The capacity of a technology is usually a continuous decision variable, which can be within the range of 0 and ``energy_cap_max`` (the maximum capacity of a technology). In this model, we introduce a unit limit on the CHP instead:

.. literalinclude:: ../../calliope/example_models/urban_scale/overrides.yaml
   :language: yaml
   :dedent: 8
   :start-after: # chp-start
   :end-before: # chp-end

A unit maximum allows a discrete, integer number of CHP to be purchased, each having a capacity of ``energy_cap_per_unit``. Any of ``energy_cap_max``, ``energy_cap_min``, or ``energy_cap_equals`` are now ignored, in favour of ``units_max``, ``units_min``, or ``units_equals``. A useful feature unlocked by introducing this is the ability to set a minimum operating capacity which is *only* enforced when the technology is operating. In the LP model, ``energy_cap_min_use`` would force the technology to operate at least at that proportion of its maximum capacity at each time step. In this model, the newly introduced ``energy_cap_min_use`` of 0.2 will ensure that the output of the CHP is 20% of its maximum capacity in any time step in which it has a non-zero output.

Purchase cost
=============

The boiler does not have a unit limit, it still utilises the continuous variable for its capacity. However, we have introduced a ``purchase`` cost:

.. literalinclude:: ../../calliope/example_models/urban_scale/overrides.yaml
   :language: yaml
   :dedent: 8
   :start-after: # boiler-start
   :end-before: # boiler-end

By introducing this, the boiler now has a binary decision variable associated with it, which is 1 if the boiler has a non-zero ``energy_cap`` (i.e. the optimisation results in investment in a boiler) and 0 if the capacity is 0. The purchase cost is applied to the binary result, providing a fixed cost on purchase of the technology, irrespective of the technology size. In physical terms, this may be associated with the cost of pipework, land purchase, etc. The purchase cost is also imposed on the CHP, which is applied to the number of integer CHP units in which the solver chooses to invest.

MILP functionality can be easily applied, but convergence is slower as a result of integer/binary variables. It is recommended to use a commercial solver (e.g. Gurobi, CPLEX) if you wish to utilise these variables outside this example model.

Running the model
=================

We now take you through running the model in a Jupyter notebook, which is included fully below. To download and run the notebook yourself, you can find it :nbviewer_docs:`here <user/tutorials/milp.ipynb>`. You will need to have Calliope installed.

.. raw:: html
   :file: tutorials/milp.html
