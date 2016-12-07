
=================
Model formulation
=================

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

--------------------------------------
Objective function (cost minimization)
--------------------------------------

Provided by: :func:`calliope.constraints.objective.objective_cost_minimization`

The default objective function minimizes cost:

.. math::

   min: z = \sum_y (weight(y) \times \sum_x cost(y, x, k=k_{m}))

where :math:`k_{m}` is the monetary cost class.

Alternative objective functions can be used by setting the ``objective`` in the model configuration (see :ref:`config_reference_model_wide`).

`weight(y)` is 1 by default, but can be adjusted to change the relative weighting of costs of different technologies in the objective, by setting ``weight`` on any technology (see :ref:`config_reference_techs`).

-----------------
Basic constraints
-----------------

Node resource
-------------

Provided by: :func:`calliope.constraints.base.node_resource`

Defines the following variables:

 * ``rs``: resource to/from storage (+ production, - consumption)
 * ``r_area``: resource collector area
 * ``rbs``: secondary resource to storage (+ production)

It also defines the constraint ``c_rs``. This constraint defines the available resource for a node, :math:`r_{avail}`:

.. math::

   r_{avail}(y, x, t) = r(y, x, t) \times r_{scale}(y, x) \times r_{area}(y, x) \times r_{eff}(y)

The ``c_rs`` constraint also decides how the resource and storage are linked.

If the option ``constraints.force_r`` is set to true, then

.. math::

   r_{s}(y, x, t) = r_{avail}(y, x, t)

If that option is not set, and the technology inherits from the ``supply`` or ``unmet_demand`` base technologies,

.. math::

   r_{s}(y, x, t) \leq r_{avail}(y, x, t)

Finally, if it inherits from the ``demand`` technology,

.. math::

   r_{s}(y, x, t) \geq r_{avail}(y, x, t)

.. Note:: For the case of ``storage`` technologies, :math:`r{s}` is forced to :math:`0` for internal reasons, while for ``transmission`` technologies, it is unconstrained. This is irrelevant when defining models and defining a resource for either ``storage`` or ``transmission`` technologies has no effect.

Node energy balance
-------------------

Provided by: :func:`calliope.constraints.base.node_energy_balance`

Defines the following variables:

* ``s``: storage level
* ``es_prod``: energy from storage to carrier
* ``es_con``: energy from carrier to storage

It also defines three constraints, which are discussed in turn:

* ``c_s_balance_pc``: energy balance for supply, demand, and storage technologies
* ``c_s_balance_transmission``: energy balance for transmission technologies
* ``c_s_balance_conversion``: energy balance for conversion technologies

Supply/demand/storage balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A node that allows storage and either supply or demand is the most complex case, with the balancing equation

.. math::

   s(y, x, t) = s_{minusone} + r_{s}(y, x, t) + r_{bs}(y, x, t) - e_{prod} - e_{con}

:math:`e_{prod}` is defined as :math:`es_{prod}(c, y, x, t) \times e_{eff}(y, x, t)`.

:math:`e_{con}` is defined as :math:`\frac{es_{con}(c, y, x, t)}{e_{eff}(y, x, t)}`, or as :math:`0` if :math:`e_{eff}(y, x, t)` is :math:`0`.

:math:`r_{bs}(y, x, t)` is the secondary resource and is always set to zero unless the technology explicitly defines a secondary resource.

:math:`s(y, x, t)` is the storage level at time :math:`t`.

:math:`s_{minusone}` describes the state of storage at the previous timestep. :math:`s_{minusone} = s_{init}(y, x)` at time :math:`t=0`. Else,

.. math::

   s_{minusone} = (1 - s_{loss}) \times timeres(t-1) \times s(y, x, t-1)

.. Note:: In operation mode, ``s_init`` is carried over from the previous optimization period.

If no storage is allowed, the balancing equation simplifies to

.. math::

   r_{s}(y, x, t) + r_{bs}(y, x, t) = e_{prod} + e_{con}

Transmission balance
^^^^^^^^^^^^^^^^^^^^

Transmission technologies are internally expanded into two technologies per transmission link, of the form ``technology_name:destination``.

For example, if the technology ``hvdc`` is defined and connects ``region_1`` to ``region_2``, the framework will internally create a technology called ``hvdc:region_2`` which exists in ``region_1`` to connect it to ``region_2``, and a technology called ``hvdc:region_1`` which exists in ``region_2`` to connect it to ``region_1``.

The balancing for transmission technologies is given by

.. math::

   es_{prod}(c, y, x, t) = -1 \times es_{con}(c, y_{remote}, x_{remote}, t) \times e_{eff}(y, x, t) \times e_{eff,perdistance}(y, x)

Here, :math:`x_{remote}, y_{remote}` are x and y at the remote end of the transmission technology. For example, for ``(y, x) = ('hvdc:region_2', 'region_1')``, the remotes would be ``('hvdc:region_1', 'region_2')``.

:math:`es_{prod}(c, y, x, t)` for ``c='power', y='hvdc:region_2', x='region_1'`` would be the import of power from ``region_2`` to ``region_1``, via a ``hvdc`` connection, at time ``t``.

This also shows that transmission technologies can have both a static or time-dependent efficiency (line loss), :math:`e_{eff}(y, x, t)`, and a distance-dependent efficiency, :math:`e_{eff,perdistance}(y, x)`.

For more detail on distance-dependent configuration see :doc:`configuration`.

Conversion balance
^^^^^^^^^^^^^^^^^^

The conversion balance is given by

.. math::

   es_{prod}(c_{prod}, y, x, t) = -1 \times es_{con}(c_{source}, y, x, t) \times e_{eff}(y, x, t)

The principle is similar to that of the transmission balance. The production of carrier :math:`c_{prod}` (the ``carrier`` option set for the conversion technology) is driven by the consumption of carrier :math:`c_{source}` (the ``source_carrier`` option set for the conversion technology).


Node build constraints
----------------------

Provided by: :func:`calliope.constraints.base.node_constraints_build`

Defines the following variables:

* ``s_cap``: installed storage capacity
* ``r_cap``: installed resource to/from storage conversion capacity
* ``e_cap``: installed storage to/from grid conversion capacity (gross)
* ``e_cap_net``: installed storage to/from grid conversion capacity (net)
* ``rb_cap``: installed secondary resource conversion capacity

Built capacity is managed by six constraints.

``c_s_cap`` constrains the built storage capacity by :math:`s_{cap}(y, x) \leq s_{cap,max}(y, xi)`. If ``y.constraints.use_s_time`` is true at location ``x``, then ``y.constraints.s_time.max`` and ``y.constraints.e_cap.max`` are used to to compute ``s_cap.max`` at reference efficiency. If ``y.constraints.s_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

``c_r_cap`` constrains the built resource conversion capacity by :math:`r_{cap}(y, x) \leq r_{cap,max}(y, x)`. If the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

``c_r_area`` constrains the resource conversion area by :math:`r_{area}(y, x) \leq r_{area,max}(y, x)`. By default, ``y.constraints.r_area.max`` is set to false, and in that case, :math:`r_{area}(y, x)` is forced to :math:`1.0`. If the model is running in operational mode, the inequality in the equation above is turned into an equality constraint. Finally, if ``y.constraints.r_area_per_e_cap`` is given, then the equation :math:`r_{area}(y, x) = e_{cap}(y, x) * r\_area\_per\_cap` applies instead.

``c_e_cap`` constrains the carrier conversion capacity. If a technology ``y`` is not allowed at a location ``x``, :math:`e_{cap}(y, x) = 0` is forced. Else, :math:`e_{cap}(y, x) \leq e_{cap,max}(y, x) \times e\_cap\_scale` applies. ``y.constraints.e_cap_scale`` defaults to 1.0 but can be set on a per-technology, per-location basis if necessary. Finally, if ``y.constraints.e_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

The ``c_e_cap_gross_net`` constraint is relevant only if ``y.constraints.c_eff`` is set to anything other than 1.0 (the default). In that case, :math:`e_{cap}(y, x) \times c_{eff} = e_{cap,net}(y, x)` computes the net installed carrier conversion capacity.

The final constraint, ``c_rb_cap``, manages the secondary resource conversion capacity by :math:`rb_{cap}(y, x) \leq rb_{cap,max}(y, x)`. If ``y.constraints.rb_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint. There is an additional relevant option, ``y.constraints.rb_cap_follows``, which can be overridden on a per-location basis. It can be set either to ``r_cap`` or ``e_cap``, and if set, sets ``c_rb_cap`` to track one of these, ie, :math:`rb_{cap,max} = r_{cap}(y, x)` (analogously for ``e_cap``), and also turns the constraint into an equality constraint.

Node operational constraints
----------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_operational`

This component ensures that nodes remain within their operational limits, by constraining ``rs``, ``es``, ``s``, and ``rbs``.

:math:`r_{s}(y, x, t)` is constrained to remain within :math:`r_{cap}(y, x)`, with the two constraints ``c_rs_max_upper`` and ``c_rs_max_lower``:

.. math::

   r_{s}(y, x, t) \leq timeres(t) \times r_{cap}(y, x)

   r_{s}(y, x, t) \geq -1 \times timeres(t) \times r_{cap}(y, x)

:math:`e_{s}(c, y, x, t)` is constrained by three constraints, ``c_es_prod_max``, ``c_es_prod_min``, and ``c_es_con_max``:

.. math::

   e_{s,prod}(c, y, x, y) \leq timeres(t) \times e_{cap}(y, x)

if ``c`` is the ``carrier`` of ``y``, else :math:`e_{s,prod}(c, y, x, y) = 0`.

If ``e_cap_min_use`` is defined, the minimum output is constrained by

.. math::

   e_{s,prod}(c, y, x, y) \geq timeres(t) \times e_{cap}(y, x) \times e_{cap,minuse}

For technologies where ``y.constraints.e_con`` is true (it defaults to false), and for conversion technologies,

.. math::

   e_{s,con}(c, y, x, y) \geq -1 \times timeres(t) \times e_{cap}(y, x)

and :math:`e_{s,con}(c, y, x, y) = 0` otherwise.

The constraint ``c_s_max`` ensures that storage cannot exceed its maximum size by

.. math::

   s(y, x, t) \leq s_{cap}(y, x)

And finally, ``c_rbs_max`` constrains the secondary resource by

.. math::

   rb_{s}(y, x, t) \leq timeres(t) \times rb_{cap}(y, x)

There is an additional check if ``y.constraints.rb_startup_only`` is true. In this case, :math:`rb_{s}(y, x, t) = 0` unless the current timestep is still within the startup time set in the ``startup_time_bounds`` model-wide setting. This can be useful to prevent undesired edge effects from occurring in the model.

Transmission constraints
------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_transmission`

This component provides a single constraint, ``c_transmission_capacity``, which forces :math:`e_{cap}` to be symmetric for transmission nodes. For example, for for a given transmission line between :math:`x_1` and :math:`x_2`, using the technology ``hvdc``:

.. math::

   e_{cap}(hvdc:x_2, x_1) = e_{cap}(hvdc:x_1, x_2)

Node parasitics
---------------

Provided by: :func:`calliope.constraints.base.node_parasitics`

Defines the following variables:

 * ``ec_prod``: storage to carrier after parasitics (positive, production)
 * ``ec_con``: carrier to storage after parasitics (negative, consumption)

There are two constraints, ``c_ec_prod`` and ``c_ec_con``, which constrain ``ec`` by

 .. math::

   ec_{prod}(c, y, x, t) = es_{prod}(c, y, x, t) \times c_{eff}(y, x)

   ec_{con}(c, y, x, t) = \frac{es_{con}(c, y, x, t)}{c_{eff}(y, x)}

For conversion and transmission technologies, the second equation reads :math:`ec_{con}(c, y, x, t) = es_{con}(c, y, x, t)` so that the internal losses are applied only once.

The two variables ``ec_prod`` and ``ec_con`` are only defined in the model for technologies where ``c_eff`` is not 1.0.

.. Note:: When reading the model solution, Calliope automatically manages the ``es`` and ``ec`` variables. In the solution, every technology has an ``ec`` variable, which is simply set to ``es`` wherever it was not defined, to make the solution consistent.

Node costs
----------

Provided by: :func:`calliope.constraints.base.node_costs`

Defines the following variables:

* ``cost``: total costs
* ``cost_con``: construction costs
* ``cost_op_fixed``: fixed operation costs
* ``cost_op_var``: variable operation costs
* ``cost_op_fuel``: primary resource fuel costs
* ``cost_op_rb``: secondary resource fuel costs

These equations compute costs per node.

Weights are adjusted for individual timesteps depending on the timestep reduction methods applied (see :ref:`run_time_res`), and are given by :math:`W(t)` when computing costs.

The depreciation rate for each cost class ``k`` is calculated as

.. math::

   d(y, k) = \frac{1}{plant\_life(y)}

if the interest rate :math:`i` is :math:`0`, else

.. math::

   d(y, k) = \frac{i \times (1 + i(y, k))^{plant\_life(k)}}{(1 + i(y, k))^{plant\_life(k)} - 1}

Costs are split into construction and operational and maintenance (O&M) costs. The total costs are computed in ``c_cost`` by

.. math::

   cost(y, x, k) = cost_{con}(y, x, k) + cost_{op,fixed}(y, x, k) + cost_{op,var}(y, x, k) + cost_{op,fuel}(y, x, k) + cost_{op,rb}(y, x, k)

The construction costs are computed in ``c_cost_con`` by

.. math::

   cost_{con}(y, x, k) &= d(y, k) \times \frac{\sum\limits_t timeres(t) \times W(t)}{8760} \\
   & \times (cost_{s\_cap}(y, k) \times s_{cap}(y, x) \\
   & + cost_{r\_cap}(y, k) \times r_{cap}(y, x) \\
   & + cost_{r\_area}(y, k) \times r_{area}(y, x) \\
   & + cost_{e\_cap}(y, k) \times e_{cap}(y, x)) \\
   & + cost_{rb\_cap}(y, k) \times rb_{cap}(y, x))

The costs are as defined in the model definition, e.g. e.g. :math:`cost_{r\_cap}(y, k)` corresponds to ``y.costs.k.r_cap``.

For transmission technologies, :math:`cost_{e\_cap}(y, k)` is computed differently, to include the per-distance costs:

.. math::

   cost_{e\_cap,transmission}(y, k) = \frac{cost_{e\_cap}(y, k) + cost_{e\_cap,perdistance}(y, k)}{2}

This implies that for transmission technologies, the cost of construction is split equally across the two locations connected by the technology.

The O&M costs are computed in four separate constraints, ``cost_op_fixed``, ``cost_op_var``, ``cost_op_fuel``, and ``cost_op_rb``, by

.. math::

   cost_{op,fixed}(y, x, k) &= cost_{om\_frac}(y, k) \times cost_{con}(y, x, k) \\
   & + cost_{om\_fixed}(y, k) \times e_{cap}(y, x) \\
   & \times \frac{\sum\limits_t timeres(t) \times W(t)}{8760}

.. math::

   cost_{op,var}(y, x, k) = cost_{om\_var}(y, k) \times \sum_t W(t) \times e_{prod}(c, y, x, t)

   cost_{op,fuel}(y, x, k) = \frac{cost_{om\_fuel}(y, k) \times \sum_t W(t) \times r_{s}(y, x, t)}{r_{eff}(y, x)}

   cost_{op,rb}(y, x, k) = \frac{cost_{om\_rb}(y, k) \times \sum_t W(t) \times r_{bs}(y, x, t)}{rb_{eff}(y, x)}


Model balancing constraints
---------------------------

Provided by: :func:`calliope.constraints.base.model_constraints`

Model-wide balancing constraints are constructed for nodes that have children. They differentiate between:

* ``c = power``
* All other ``c``

In the first case, the following balancing equation applies:

.. math::

   \sum_{y, x \in X_{i}} ec_{prod}(c=c_{p}, y, x, t) + \sum_{y, x \in X_{i}} ec_{con}(c=c_{p}, y, x, t) = 0 \qquad\forall i, t

:math:`i` are the level 0 locations, and :math:`X_{i}` is the set of level 1 locations (:math:`x`) within the given level 0 location, together with that location itself. :math:`c` is the carrier, and :math:`c_{p}` the carrier for power.

For ``c`` other than ``power``, the balancing equation is as above, but with a :math:`\geq` inequality, and the corresponding change to :math:`c`.

.. Note:: The actual balancing constraint is implemented such that ``es`` and ``ec`` are used in the sum as appropriate for each technology.

--------------------
Planning constraints
--------------------

These constraints are loaded automatically, but only when running in planning mode.

.. _system_margin:

System margin
-------------

Provided by: :func:`calliope.constraints.planning.system_margin`

This is a simplified capacity margin constraint, requiring the capacity to supply a given carrier in the time step with the highest demand for that carrier to be above the demand in that timestep by at least the given fraction:

.. math::

   \sum_y \sum_x es_{prod}(c, y, x, t_{max,c}) \times (1 + m_{c}) \leq timeres(t) \times \sum_{y_{c}} \sum_x (e_{cap}(y, x) / e_{eff,ref}(y, x))

where :math:`y_{c}` is the subset of ``y`` that delivers the carrier ``c`` and :math:`m_{c}` is the system margin for that carrier.

For each carrier (with the name ``carrier_name``), Calliope attempts to read the model-wide option ``system_margin.carrier_name``, only applying this constraint if a setting exists.

.. _optional_constraints:

--------------------
Optional constraints
--------------------

Optional constraints are included with Calliope but not loaded by default (see the :ref:`configuration section <loading_optional_constraints>` for instructions on how to load them in a model).

These optional constraints can be used both in planning and operational modes.

Ramping
-------

Provided by: :func:`calliope.constraints.optional.ramping_rate`

Constrains the rate at which plants can adjust their output, for technologies that define ``constraints.e_ramping``:

.. math::

   diff = \frac{es_{prod}(c, y, x, t) + es_{con}(c, y, x, t)}{timeres(t)} - \frac{es_{prod}(c, y, x, t-1) + es_{con}(c, y, x, t-1)}{timeres(t-1)}

   max\_ramping\_rate = e_{ramping} \times e_{cap}(y, x)

   diff \leq max\_ramping\_rate

   diff \geq -1 \times max\_ramping\_rate

.. _group_fraction:

Group fractions
---------------

Provided by: :func:`calliope.constraints.optional.group_fraction`

This component provides the ability to constrain groups of technologies to provide a certain fraction of total output, a certain fraction of total capacity, or a certain fraction of peak power demand. See :ref:`config_parents_and_groups` in the configuration section for further details on how to set up groups of technologies.

The settings for the group fraction constraints are read from the model-wide configuration, in a ``group_fraction`` setting, as follows:

.. code-block:: yaml

   group_fraction:
      capacity:
         renewables: ['>=', 0.8]

This is a minimal example that forces at least 80% of the installed capacity to be renewables. To activate the output group constraint, the ``output`` setting underneath ``group_fraction`` can be set in the same way, or ``demand_power_peak`` to activate the fraction of peak power demand group constraint.

.. TODO ignored_techs option

For the above example, the ``c_group_fraction_capacity`` constraint sets up an equation of the form

.. math::

   \sum_{y^*} \sum_x e_{cap}(y, x) \geq fraction \times \sum_y \sum_x e_{cap}(y, x)

Here, :math:`y^*` is the subset of :math:`y` given by the specified group, in this example, ``renewables``. :math:`fraction` is the fraction specified, in this example, :math:`0.8`. The relation between the right-hand side and the left-hand side, :math:`\geq`, is determined by the setting given, ``>=``, which can be ``==``, ``<=``, or ``>=``.

If more than one group is listed under ``capacity``, several analogous constraints are set up.

Similarly, ``c_group_fraction_output`` sets up constraints in the form of

.. math::

   \sum_{y^*} \sum_x \sum_t es_{prod}(c, y, x, t) \geq fraction \times \sum_y \sum_x \sum_t es_{prod}(c, y, x, t)

Finally, ``c_group_fraction_demand_power_peak`` sets up constraints in the form of

.. math::

   \sum_{y^*} \sum_x e_{cap}(y, x) \geq fraction \times (-1 - m_{c}) \times peak

   peak = \frac{\sum_x r(y_d, x, t_{peak}) \times r_{scale}(y_d, x)}{timeres(t_{peak})}

This assumes the existence of a technology, ``demand_power``, which defines a demand (negative resource). :math:`y_d` is ``demand_power``. :math:`m_{c}` is the capacity margin defined for the carrier ``c`` in the model-wide settings (see :ref:`system_margin`). :math:`t_{peak}` is the timestep where :math:`r(y_d, x, t)` is maximal.

Whether any of these equations are equalities, greater-or-equal-than inequalities, or lesser-or-equal-than inequalities, is determined by whether ``>=``, ``<=``, or ``==`` is given in their respective settings.
