
=================
Model formulation
=================

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

.. _time_varying_vs_constant_parameters:

------------------------------------------
Time-varying vs. constant model parameters
------------------------------------------

Some model parameters which are defined over the set of time steps ``t`` can either given as time series or as constant values. If given as constant values, the same value is used for each time step ``t``. For details on how to define a parameter as time-varying and how to load time series data into it, see the :ref:`time series description in the model configuration section <configuration_timeseries>`.

------------------
Decision variables
------------------

Capacity
--------

* ``s_cap(y, x)``: installed storage capacity. Supply plus/Storage only
* ``r_cap(y, x)``: installed resource <-> storage conversion capacity
* ``e_cap(y, x)``: installed storage <-> grid conversion capacity (gross)
* ``r2_cap(y, x)``: installed secondary resource conversion capacity
* ``r_area(y, x)``: resource collector area

Unit Commitment
---------------

* ``r(y, x, t)``: resource <-> storage/carrier_in (+ production, - consumption)
* ``r2(y, x, t)``: secondary resource -> storage (+ production)
* ``c_prod(c, y, x, t)``: resource/storage/carrier_in -> carrier_out (+ production)
* ``c_con(c, y, x, t)``: resource/storage/carrier_in <- carrier_out (- consumption)
* ``s(y, x, t)``: total energy stored in device
* ``export(y, x, t)``: carrier_out -> export

Costs
-----

* ``cost(y, x, k)``: total costs
* ``cost_fixed(y, x, k)``: fixed operation costs
* ``cost_var(y, x, k, t)``: variable operation costs

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

Defines constraint c_r_available:

.. math::

   r_{avail}(y, x, t) = resource(y, x, t) \times r_{scale}(y, x) \times r_{area}(y, x)

Which limits the resource flow **to** ``supply`` and ``supply_plus`` technologies, or **from** ``demand`` technologies.

For ``supply``:

If the option ``constraints.force_r`` is set to true, then

.. math::

   \frac{c_{prod}(c, y, x, t)}{e_{eff}(y, x, t)} = r_{avail}(y, x, t)

If that option is not set:

.. math::

    \frac{c_{prod}(c, y, x, t)}{e_{eff}(y, x, t)} \leq r_{avail}(y, x, t)

For ``demand``:

If the option ``constraints.force_r`` is set to true, then

.. math::

   c_{con}(c, y, x, t) \times e_{eff}(y, x, t) = r_{avail}(y, x, t)

If that option is not set:

.. math::

  c_{con}(c, y, x, t) \times e_{eff}(y, x, t) \geq r_{avail}(y, x, t)

For ``supply_plus``:

If the option ``constraints.force_r`` is set to true, then

.. math::

   r(y, x, t) = r_{avail}(y, x, t) \times r_{eff}(y, x, t)

If that option is not set:

.. math::

  r(y, x, t) \leq r_{avail}(y, x, t) \times r_{eff}(y, x, t)

.. Note:: For all other technology types, defining a resource is irrelevant, so they are not constrained here.

Node energy balance
-------------------

Provided by: :func:`calliope.constraints.base.node_energy_balance`

Defines nine constraints, which are discussed in turn:

* ``c_balance_transmission``: energy balance for ``transmission`` technologies
* ``c_balance_conversion``: energy balance for ``conversion`` technologies
* ``c_balance_conversion_plus``: energy balance for ``conversion_plus`` technologies
* ``c_balance_conversion_plus_secondary_out``: energy balance for ``conversion_plus`` technologies which have a secondary output carriers
* ``c_balance_conversion_plus_tertiary_out``: energy balance for ``conversion_plus`` technologies which have a tertiary output carriers
* ``c_balance_conversion_plus_secondary_in``: energy balance for ``conversion_plus`` technologies which have a secondary input carriers
* ``c_balance_conversion_plus_tertiary_in``: energy balance for ``conversion_plus`` technologies which have a tertiary input carriers
* ``c_balance_supply_plus``: energy balance for ``supply_plus`` technologies
* ``c_balance_storage``: energy balance for ``storage`` technologies

Transmission balance
^^^^^^^^^^^^^^^^^^^^

Transmission technologies are internally expanded into two technologies per transmission link, of the form ``technology_name:destination``.

For example, if the technology ``hvdc`` is defined and connects ``region_1`` to ``region_2``, the framework will internally create a technology called ``hvdc:region_2`` which exists in ``region_1`` to connect it to ``region_2``, and a technology called ``hvdc:region_1`` which exists in ``region_2`` to connect it to ``region_1``.

The balancing for transmission technologies is given by

.. math::

   c_{prod}(c, y, x, t) = -1 \times c_{con}(c, y_{remote}, x_{remote}, t) \times e_{eff}(y, x, t) \times e_{eff,perdistance}(y, x)

Here, :math:`x_{remote}, y_{remote}` are x and y at the remote end of the transmission technology. For example, for ``(y, x) = ('hvdc:region_2', 'region_1')``, the remotes would be ``('hvdc:region_1', 'region_2')``.

:math:`c_{prod}(c, y, x, t)` for ``c='power', y='hvdc:region_2', x='region_1'`` would be the import of power from ``region_2`` to ``region_1``, via a ``hvdc`` connection, at time ``t``.

This also shows that transmission technologies can have both a static or time-dependent efficiency (line loss), :math:`e_{eff}(y, x, t)`, and a distance-dependent efficiency, :math:`e_{eff,perdistance}(y, x)`.

For more detail on distance-dependent configuration see :doc:`configuration`.

Conversion balance
^^^^^^^^^^^^^^^^^^

The conversion balance is given by

.. math::

   c_{prod}(c_{out}, y, x, t) = -1 \times c_{con}(c_{in}, y, x, t) \times e_{eff}(y, x, t)

The principle is similar to that of the transmission balance. The production of carrier :math:`c_{out}` (the ``carrier_out`` option set for the conversion technology) is driven by the consumption of carrier :math:`c_{in}` (the ``carrier_in`` option set for the conversion technology).

Conversion_plus balance
^^^^^^^^^^^^^^^^^^^^^^^

Conversion plus technologies can have several carriers in and several carriers out, leading to a more complex production/consumption balance.

For the primary carrier(s), the balance is:

.. math::

  \sum\limits_{c_{out_1}} \frac{c_{prod}(c_{out_1}, y, x, t) }{carrier_{fraction}(c_{out_1})} =  -1 \times \sum\limits_{c_{in_1}} c_{con}(c_{in_1}, y, x, t) \times carrier_{fraction}(c_{in_1}) \times e_{eff}(x, y, t)

Where ``c_{out_1}`` and ``c_{in_1}`` are the sets of primary production and consumption carriers, respectively and ``carrier_{fraction}`` is the relative contribution of these carriers, as defined in ??.

The remaining constraints (``c_balance_conversion_plus_secondary_out``, ``c_balance_conversion_plus_tertiary_out``, ``c_balance_conversion_plus_secondary_in``, ``c_balance_conversion_plus_tertiary_in``) link the input/output of the technology secondary and tertiary carriers to the primary consumption/production.

For production:

.. math::

  \sum\limits_{c_{out_1}} \frac{c_{prod}}{\frac{(c_{out_1}, y, x, t)}{carrier_{fraction}(c_{out_1})}} \times min(carrier_{fraction}(c_{out_x}))=  \sum\limits_{c_{out_x}} c_{prod}(c_{out_x}, y, x, t) \times \frac{carrier_{fraction}(c_{out_x})}{min(carrier_{fraction}(c_{out_x}))}

For consumption:

.. math::

  \sum\limits_{c_{in_1}} \frac{c_{con}(c_{in_1}, y, x, t) }{carrier_{fraction}(c_{in_1})} \times min(carrier_{fraction}(c_{in_x}))=  \sum\limits_{c_{in_x}} c_{con}(c_{in_x}, y, x, t) \times \frac{carrier_{fraction}(c_{in_x})}{min(carrier_{fraction}(c_{in_x}))}

Where ``x`` is either 2 (secondary carriers) or 3 (tertiary carriers).

.. Warning::

   The ``conversion_plus`` technology is still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior. It is also possible to use a combination of several regular ``conversion`` technologies to achieve some of the behaviors covered by ``conversion_plus``, but at the expense of model complexity.

Supply_plus balance
^^^^^^^^^^^^^^^^^^^

``Supply_plus`` technologies are ``supply`` technologies with more control over resource flow. You can have multiple resources, a resource capacity, and storage of resource before it is converted to the primary carrier_out.

If storage is possible:

.. math::

   s(y, x, t) = s_{minusone} + r(y, x, t) + r_{2}(y, x, t) - c_{prod}

Otherwise:

.. math::

  r(y, x, t) = c_{prod} - r_{2}


Where:

:math:`c_{prod}` is defined as :math:`\frac{c_{prod}(c, y, x, t)}{total_{eff}}`.

:math:`total_{eff}(y, x, t)` is defined as :math:`e_{eff}(y, x, t) + p_{eff}(y, x, t)`, the plant efficiency including parasitic losses

:math:`r_{2}(y, x, t)` is the secondary resource and is always set to zero unless the technology explicitly defines a secondary resource.

:math:`s(y, x, t)` is the storage level at time :math:`t`.

:math:`s_{minusone}` describes the state of storage at the previous timestep. :math:`s_{minusone} = s_{init}(y, x)` at time :math:`t=0`. Else,

.. math::

   s_{minusone} = (1 - s_{loss}) \times timeres(t-1) \times s(y, x, t-1)

.. Note:: In operation mode, ``s_init`` is carried over from the previous optimization period.


Storage balance
^^^^^^^^^^^^^^^^^^^^
Storage technologies balance energy charge, energy discharge, and energy stored:

.. math::

   s(y, x, t) = s_{minusone} - c_{prod} - c_{con}

Where:

:math:`c_{prod}` is defined as :math:`\frac{c_{prod}(c, y, x, t)}{total_{eff}}` if :math:`total_{eff} > 0`, otherwise :math:`c_{prod} = 0`

:math:`c_{con}` is defined as :math:`c_{con}(c, y, x, t) \times total_{eff}`

:math:`total_{eff}(y, x, t)` is defined as :math:`e_{eff}(y, x, t) + p_{eff}(y, x, t)`, the plant efficiency including parasitic losses

:math:`s(y, x, t)` is the storage level at time :math:`t`.

:math:`s_{minusone}` describes the state of storage at the previous timestep. :math:`s_{minusone} = s_{init}(y, x)` at time :math:`t=0`. Else,

.. math::

   s_{minusone} = (1 - s_{loss}) \times timeres(t-1) \times s(y, x, t-1)

.. Note:: In operation mode, ``s_init`` is carried over from the previous optimization period.


Node build constraints
----------------------

Provided by: :func:`calliope.constraints.base.node_constraints_build`

Built capacity is managed by six constraints.

``c_s_cap``
^^^^^^^^^^^
This constrains the built storage capacity by:

.. math::

    s_{cap}(y, x) \leq s_{cap,max}(y, x)

If ``y.constraints.s_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

If both :math:`e_{cap,max}(y, x)` and :math:`charge\_rate` are not given, :math:`s_{cap}(y, x)` is automatically set to zero.

If ``y.constraints.s_time.max`` is true at location ``x``, then ``y.constraints.s_time.max`` and ``y.constraints.e_cap.max`` are used to to compute ``s_cap.max``. The minimum value of ``s_cap.max`` is taken, based on analysis of all possible time sets which meet the s_time.max value. This allows time-varying efficiency, :math:`e_{eff}(y, x, t)` to be accounted for.

``c_r_cap``
^^^^^^^^^^^
This constrains the built resource conversion capacity by:

.. math::

  r_{cap}(y, x) \leq r_{cap,max}(y, x)

If the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

``c_r_area``
^^^^^^^^^^^^
This constrains the resource conversion area by:

.. math::

  r_{area}(y, x) \leq r_{area,max}(y, x)

By default, ``y.constraints.r_area.max`` is set to false, and in that case, :math:`r_{area}(y, x)` is forced to :math:`1.0`. If the model is running in operational mode, the inequality in the equation above is turned into an equality constraint. Finally, if ``y.constraints.r_area_per_e_cap`` is given, then the equation :math:`r_{area}(y, x) = e_{cap}(y, x) * r\_area\_per\_cap` applies instead.

``c_e_cap``
^^^^^^^^^^^
This constrains the carrier conversion capacity by:

.. math::
  e_{cap}(y, x) \leq e_{cap,max}(y, x) \times e\_cap\_scale

If a technology ``y`` is not allowed at a location ``x``, :math:`e_{cap}(y, x) = 0` is forced.

``y.constraints.e_cap_scale`` defaults to 1.0 but can be set on a per-technology, per-location basis if necessary.

If ``y.constraints.e_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

``c_e_cap_storage``
^^^^^^^^^^^^^^^^^^^
This constrains the carrier conversion capacity for storage technologies by:

.. math::
  e_{cap}(y, x) \leq e_{cap,max}

Where :math:`e_{cap,max} = s_{cap}(y, x) * charge\_rate * e\_cap\_scale`

``y.constraints.e_cap_scale`` defaults to 1.0 but can be set on a per-technology, per-location basis if necessary.

``c_r2_cap``
^^^^^^^^^^^^
This manages the secondary resource conversion capacity by:

.. math::
  r2_{cap}(y, x) \leq r2_{cap,max}(y, x)

If ``y.constraints.r2_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

There is an additional relevant option, ``y.constraints.r2_cap_follows``, which can be overridden on a per-location basis. It can be set either to ``r_cap`` or ``e_cap``, and if set, sets ``c_r2_cap`` to track one of these, ie, :math:`r2_{cap,max} = r_{cap}(y, x)` (analogously for ``e_cap``), and also turns the constraint into an equality constraint.

Node operational constraints
----------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_operational`

This component ensures that nodes remain within their operational limits, by constraining ``r``, ``c_prod``, ``c_con``, ``s``, ``r2``, and ``export``.

``r``
^^^^^
:math:`r(y, x, t)` is constrained to remain within :math:`r_{cap}(y, x)`, with the constraint ``c_r_max_upper``:

.. math::

   r(y, x, t) \leq time\_res(t) \times r_{cap}(y, x)

``c_prod``
^^^^^^^^^^
:math:`c_prod(c, y, x, t)` is constrained by ``c_prod_max`` and ``c_prod_min``:

.. math::

   c_{prod}(c, y, x, t) \leq time\_res(t) \times e_{cap}(y, x) \times p_{eff}(y, x, t)

if ``c`` is the ``carrier_out`` of ``y``, else :math:`c_{prod}(c, y, x, y) = 0`.

If ``e_cap_min_use`` is defined, the minimum output is constrained by:

.. math::

   c_{prod}(c, y, x, t) \geq time\_res(t) \times e_{cap}(y, x) \times e_{cap,minuse} \times p_{eff}(y, x, t)

These contraints are skipped for ``conversion_plus`` technologies if ``c`` is not the primary carrier.

``c_con``
^^^^^^^^^
For technologies which are not ``supply`` or ``supply_plus``, :math:`c_con(c, y, x, t)` is non-zero. Thus :math:`c_con(c, y, x, t)` is constrainted by ``c_con_max``:

.. math::

   c_{con}(c, y, x, t) \geq -1 \times timeres(t) \times e_{cap}(y, x)

and :math:`c_{con}(c, y, x, t) = 0` otherwise.

This constraint is skipped for a ``conversion_plus`` and ``conversion`` technologies If ``c`` is a possible consumption carrier (primary, secondary, or tertiary).

``s``
^^^^^
The constraint ``c_s_max`` ensures that storage cannot exceed its maximum size by

.. math::

   s(y, x, t) \leq s_{cap}(y, x)

``r2``
^^^^^^

``c_r2_max`` constrains the secondary resource by

.. math::

   r2(y, x, t) \leq timeres(t) \times r2_{cap}(y, x)

There is an additional check if ``y.constraints.r2_startup_only`` is true. In this case, :math:`r2(y, x, t) = 0` unless the current timestep is still within the startup time set in the ``startup_time_bounds`` model-wide setting. This can be useful to prevent undesired edge effects from occurring in the model.

``export``
^^^^^^^^^^

``c_export_max`` constrains the export of a produced carrier by

.. math::

   export(y, x, t) \leq export_{cap}(y, x)

Transmission constraints
------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_transmission`

This component provides a single constraint, ``c_transmission_capacity``, which forces :math:`e_{cap}` to be symmetric for transmission nodes. For example, for for a given transmission line between :math:`x_1` and :math:`x_2`, using the technology ``hvdc``:

.. math::

   e_{cap}(hvdc:x_2, x_1) = e_{cap}(hvdc:x_1, x_2)

Node costs
----------

Provided by: :func:`calliope.constraints.base.node_costs`

These equations compute costs per node.

Weights are adjusted for individual timesteps depending on the timestep reduction methods applied (see :ref:`run_time_res`), and are given by :math:`W(t)` when computing costs.

The depreciation rate for each cost class ``k`` is calculated as

.. math::

   d(y, k) = \frac{1}{plant\_life(y)}

if the interest rate :math:`i` is :math:`0`, else

.. math::

   d(y, k) = \frac{i \times (1 + i(y, k))^{plant\_life(k)}}{(1 + i(y, k))^{plant\_life(k)} - 1}

Costs are split into fixed and variable costs. The total costs are computed in ``c_cost`` by

.. math::

   cost(y, x, k) = cost_{fixed}(y, x, k) + \sum\limits_t cost_{var}(y, x, k, t)

The fixed costs include construction costs, annual operation and maintenance (O\&M) costs, and O\&M costs which are a fraction of the construction costs.
The total fixed costs are computed in ``c_cost_fixed`` by

.. math::

  cost_{fixed}(y, x, k) = cost_{con} + cost_{om, frac} \times cost_{con} + cost_{om, fixed} \times e_{cap}(y, x) \times \frac{\sum\limits_t timeres(t) \times W(t)}{8760}

Where

.. math::

   cost_{con} &= d(y, k) \times \frac{\sum\limits_t timeres(t) \times W(t)}{8760} \\
   & \times (cost_{s\_cap}(y, k) \times s_{cap}(y, x) \\
   & + cost_{r\_cap}(y, k) \times r_{cap}(y, x) \\
   & + cost_{r\_area}(y, k) \times r_{area}(y, x) \\
   & + cost_{e\_cap}(y, k) \times e_{cap}(y, x)) \\
   & + cost_{r2\_cap}(y, k) \times r2_{cap}(y, x))

The costs are as defined in the model definition, e.g. :math:`cost_{r\_cap}(y, k)` corresponds to ``y.costs.k.r_cap``.

For transmission technologies, :math:`cost_{e\_cap}(y, k)` is computed differently, to include the per-distance costs:

.. math::

   cost_{e\_cap,transmission}(y, k) = \frac{cost_{e\_cap}(y, k) + cost_{e\_cap,perdistance}(y, k)}{2}

This implies that for transmission technologies, the cost of construction is split equally across the two locations connected by the technology.

The variable costs are O&M costs applied at each time step:

.. math::

   cost_{var} = cost_{op,var} + cost_{op,fuel} + cost_{op,r2} + cost_{op, export}

Where:

.. math::
   cost_{op,var} = cost_{om\_var}(k, y, x, t) \times \sum_t W(t) \times c_{prod}(c, y, x, t)

   cost_{op,fuel} = \frac{cost_{om\_fuel}(k, y, x, t) \times \sum_t W(t) \times r(y, x, t)}{r_{eff}(y, x)}

   cost_{op,r2} = \frac{cost_{om\_r2}(k, y, x, t) \times \sum_t W(t) \times r_{2}(y, x, t)}{r2_{eff}(y, x)}

   cost_{op, export} = cost_{export}(k, y, x, t) \times export(y, x, t)

If :math:`cost_{om\_fuel}(k, y, x, t)` is given for a ``supply`` technology and :math:`e_{eff}(y, x) > 0` for that technology, then:

.. math::
  cost_{op,fuel} =cost_{om\_fuel}(k, y, x, t) \times \sum_t W(t) \times \frac{c_{prod}(c, y, x, t)}{e_{eff}(y, x)}

``c`` is the technology primary ``carrier_out`` in all cases.


Model balancing constraints
---------------------------

Provided by: :func:`calliope.constraints.base.model_constraints`

Model-wide balancing constraints are constructed for nodes that have children:

.. math::

   \sum_{y, x \in X_{i}} c_{prod}(c, y, x, t) + \sum_{y, x \in X_{i}} c_{con}(c, y, x, t) = 0 \qquad\forall i, t

:math:`i` are the level 0 locations, and :math:`X_{i}` is the set of level 1 locations (:math:`x`) within the given level 0 location, together with that location itself.

There is also the need to ensure that technologies cannot export more energy than they produce:

.. math::

   c_{prod}(c, y, x, t) \geq export(y, x, t)

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

   \sum_y \sum_x c_{prod}(c, y, x, t_{max,c}) \times (1 + m_{c}) \leq timeres(t) \times \sum_{y_{c}} \sum_x (e_{cap}(y, x) / e_{eff}(y, x, t_{max,c}))

where :math:`y_{c}` is the subset of ``y`` that delivers the carrier ``c`` and :math:`m_{c}` is the system margin for that carrier.

For each carrier (with the name ``carrier_name``), Calliope attempts to read the model-wide option ``system_margin.carrier_name``, only applying this constraint if a setting exists.

.. _system_e_cap:

System-wide capacity
--------------------

Provided by: :func:`calliope.constraints.planning.node_constraints_build_total`

This constraint sets a maximum for capacity, ``e_cap``, across all locations for any given technology:

.. math::

  \sum_x e_{cap}(x, y) \leq e_{cap,total\_max}(y)

If :math:`e_{cap,total\_equals}` is given instead, this becomes :math:`\sum_x e_{cap}(x, y) \leq e_{cap,total\_max}(y)`.

.. math::

   \sum_y \sum_x c_{prod}(c, y, x, t_{max,c}) \times (1 + m_{c}) \leq timeres(t) \times \sum_{y_{c}} \sum_x (e_{cap}(y, x) / e_{eff}(y, x, t_{max,c}))

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

   diff = \frac{c_{prod}(c, y, x, t) + c_{con}(c, y, x, t)}{timeres(t)} - \frac{c_{prod}(c, y, x, t-1) + c_{con}(c, y, x, t-1)}{timeres(t-1)}

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

   \sum_{y^*} \sum_x \sum_t c_{prod}(c, y, x, t) \geq fraction \times \sum_y \sum_x \sum_t c_{prod}(c, y, x, t)

Finally, ``c_group_fraction_demand_power_peak`` sets up constraints in the form of

.. math::

   \sum_{y^*} \sum_x e_{cap}(y, x) \geq fraction \times (-1 - m_{c}) \times peak

   peak = \frac{\sum_x r(y_d, x, t_{peak}) \times r_{scale}(y_d, x)}{timeres(t_{peak})}

This assumes the existence of a technology, ``demand_power``, which defines a demand (negative resource). :math:`y_d` is ``demand_power``. :math:`m_{c}` is the capacity margin defined for the carrier ``c`` in the model-wide settings (see :ref:`system_margin`). :math:`t_{peak}` is the timestep where :math:`r(y_d, x, t)` is maximal.

Whether any of these equations are equalities, greater-or-equal-than inequalities, or lesser-or-equal-than inequalities, is determined by whether ``>=``, ``<=``, or ``==`` is given in their respective settings.

Available area
--------------

Provided by: :func:`calliope.constraints.optional.max_r_area_per_loc`

Where several technologies require space to acquire resource (e.g. solar collecting technologies) at a given location, this constraint provides the ability to limit the total area available at a location:

.. math::

  area_{available}(x) \geq \sum_y \sum_{xi} r_{area}(y, xi)

Where ``xi`` is the set of locations within the family tree, descending from and including ``x``.
