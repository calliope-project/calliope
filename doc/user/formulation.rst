
=================
Model formulation
=================

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

.. _time_varying_vs_constant_parameters:

------------------------------------------
Time-varying vs. constant model parameters
------------------------------------------

Some model parameters which are defined over the set of time steps ``timesteps`` can either given as time series or as constant values. If given as constant values, the same value is used for each time step ``timestep``. For details on how to define a parameter as time-varying and how to load time series data into it, see the :ref:`time series description in the model configuration section <configuration_timeseries>`.

------------------
Decision variables
------------------

Capacity
--------

* ``storage_cap(loc::tech)``: installed storage capacity. Supply plus/Storage only
* ``resource_cap(loc::tech)``: installed resource <-> storage/carrier_in conversion capacity
* ``energy_cap(loc::tech)``: installed resource/storage/carrier_in <-> carrier_out conversion capacity (gross)
* ``resource_area(loc::tech)``: resource collector area

Unit Commitment
---------------

* ``resource(loc::tech, timestep)``: resource <-> storage/carrier_in (+ production, - consumption)
* ``carrier_prod(loc::tech::carrier, timestep)``: resource/storage/carrier_in -> carrier_out (+ production)
* ``carrier_con(loc::tech::carrier, timestep)``: resource/storage/carrier_in <- carrier_out (- consumption)
* ``storage(loc::tech, timestep)``: total energy stored in technology
* ``carrier_export(loc::tech::carrier, timestep)``: carrier_out -> export

Costs
-----

* ``cost(loc::tech, cost)``: total costs
* ``cost_investment(loc::tech, cost)``: investment operation costs
* ``cost_var(loc::tech, cost, timestep)``: variable operation costs

Binary/Integer variables
------------------------

* ``units(loc::tech)``: Number of integer installed technologies
* ``purchased(loc::tech)``: Binary switch indicating whether a technology has been installed
* ``operating_units(loc::tech, timestep)``: Binary switch indicating whether a technology that has been installed is operating

--------------------------------------
Objective function (cost minimization)
--------------------------------------

Provided by: :func:`calliope.constraints.objective.objective_cost_minimization`

The default objective function minimizes cost:

.. math::

   min: z = \sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{m}))

where :math:`cost_{m}` is the monetary cost class.

Alternative objective functions can be used by setting the ``objective`` in the model configuration (see :ref:`config_reference_model_wide`).

`weight(tech)` is 1 by default, but can be adjusted to change the relative weighting of costs of different technologies in the objective, by setting ``weight`` on any technology (see :ref:`config_reference_techs`).

-----------------
Basic constraints
-----------------

Energy Balance
--------------

For all technologies, in all locations, energy in must balance with energy out (minus efficiency losses). These constraints are provided in: :func:`calliope.backend.pyomo.constraints.energy_balance.py`

1. ``system_balance_constraint_rule``
System balance ensures that, within each location, the production, consumption, and export of each carrier is balanced.
.. math::

  \sum_{loc::tech::carrier_{prod} in loc::carriers_i} carrier_{prod}(loc::tech::carrier, timestep) + \sum_{loc::tech::carrier_{con} in loc::carriers_i} carrier_{con}(loc::tech::carrier, timestep)  + \sum_{loc::tech::carrier_{export} in loc::carriers_i} carrier_{export}(loc::tech::carrier, timestep) \qquad\forall i, timesteps

Where loc::carriers is the set of all location::carrier combinations. ``carrier_export`` is ignored entirely in this constraint if there are no technologies exporting energy.

2. ``balance_supply_constraint_rule``
Limit production from supply techs to their available resource.

.. math::

  min_use(loc::tech) \times resource_{available}(loc::tech, timestep)\greq \fraq(carrier_{prod}(loc::tech::carrier, timestep))(energy_{eff}) \leq resource_{available}(loc::tech, timestep) \forall loc::tech in locs::techs_{supply}, timesteps

Where:

.. math::

   resource_{available}(loc::tech, timestep) = resource(loc::tech, timestep) \times resource_{scale}(loc::tech) \times resource_{area}(loc::tech)

If ``force_resource(loc::tech)`` is set, then the constraint becomes:

.. math::

  \fraq(carrier_{prod}(loc::tech::carrier, timestep))(energy_{eff}) \equals resource_{available}(loc::tech, timestep) \forall loc::tech in locs::techs_{supply}, timesteps

3. ``balance_demand_constraint_rule``
Limit consumption from demand techs to their required resource.

.. math::

  carrier_{con}(loc::tech::carrier, timestep) \times energy_{eff} \greq resource_{required}(loc::tech, timestep) \forall loc::tech in locs::techs_{demand}, timesteps

Where:

.. math::

   resource_{required}(loc::tech, timestep) = resource(loc::tech, timestep) \times resource_{scale}(loc::tech) \times resource_{area}(loc::tech)

If ``force_resource(loc::tech)`` is set, then the constraint becomes:

.. math::

  carrier_{con}(loc::tech::carrier, timestep) \times energy_{eff} \equals resource_{required}(loc::tech, timestep) \forall loc::tech in locs::techs_{demand}, timesteps

4. ``resource_availability_supply_plus_constraint_rule``
Limit production from supply_plus techs to their available resource.

.. math::

  resource_{con}(loc::tech, timestep) \leq resource_{available}(loc::tech, timestep) \forall loc::tech in locs::techs_{supply_plus}, timesteps

Where:

.. math::

   resource_{available}(loc::tech, timestep) = resource(loc::tech, timestep) \times resource_{scale}(loc::tech) \times resource_{area}(loc::tech) \times resource_eff(loc::tech, timestep)

If ``force_resource(loc::tech)`` is set, then the constraint becomes:

.. math::

  resource_{con}(loc::tech, timestep) \equals resource_{available}(loc::tech, timestep) \forall loc::tech in locs::techs_{supply_plus}, timesteps

5. ``balance_transmission_constraint_rule``
Balance carrier production and consumption of transmission technologies.

.. math::

  - carrier_{con}(loc_{from}::tech:loc_{to}::carrier, timestep) \times energy_{eff} \equals carrier_{prod}(loc_{to}::tech:loc_{from}::carrier, timestep) \times energy_{eff} \forall loc::tech:loc in locs::techs:locs_{transmission}, timesteps


6. ``balance_supply_plus_constraint_rule``
Balance carrier production and resource consumption of supply_plus technologies alongside any use of resource storage.

.. math::

  storage(loc::tech, timestep) = storage(loc::tech, timestep_{previous}) \times (1 - storage_{loss})^{timestep_{resolution}} \plus resource_{con}(loc::tech, timestep) - \fraq(carrier_{prod}(loc::tech::carrier, timestep))(energy_{eff} \times parasitic_{eff})

If no storage is defined for the technology, this reduces to:

.. math::

resource_{con}(loc::tech, timestep) = \fraq(carrier_{prod}(loc::tech::carrier, timestep))(energy_{eff} \times parasitic_{eff})

7. ``balance_storage_constraint_rule``
Balance carrier production and consumption of storage technologies, alongside any use of the stored volume.

.. math::

  storage(loc::tech, timestep) = storage(loc::tech, timestep_{previous}) \times (1 - storage_{loss})^{timestep_{resolution}} - carrier_{con}(loc::tech::carrier, timestep) \times energy_{eff} - \fraq(carrier_{prod}(loc::tech::carrier, timestep))(energy_{eff})

Capacity
--------
Constrain the capacity decision variables to maximum/minimum/equals the input parameters given
:func:`calliope.backend.pyomo.constraints.capacity.py`

1. ``storage_capacity_constraint_rule``
Set maximum storage capacity for supply_plus & storage techs only. This can be set by either storage_cap (kWh) or by energy_cap (charge/discharge capacity) * charge rate. If storage_cap_equals and energy_cap_equals are set for the technology, then storage_cap * charge rate = energy_cap must hold. Otherwise, take the lowest capacity defined by storage_cap_max or energy_cap_max / charge rate.

.. math::

  storage_{cap}(loc::tech) \leq storage_{cap, equals}(loc::tech)

if :math:`storage_{cap, equals}(loc::tech)` exists

else:

.. math::

  storage_{cap}(loc::tech) \leq energy_{cap, equals}(loc::tech) \times charge_{rate}`

if :math:`energy_{cap, equals}(loc::tech)` and :math:`charge_{rate}(loc::tech)` exist.

else:

.. math::

  storage_cap(loc::tech) \leq storage_{cap, max}(loc::tech)

if :math:`storage_{cap, max}(loc::tech) \leq energy_{cap, max}(loc::tech) \times charge_{rate}`.

else:

.. math::

  storage_{cap}(loc::tech) \leq energy_{cap, max}(loc::tech) \times charge_{rate}`

if :math:`energy_{cap, max}(loc::tech)` and :math:`charge_{rate}(loc::tech)` exist.

Otherwise, no maximum capacity is placed on storage.

2. ``energy_capacity_storage_constraint_rule``
Set an additional energy capacity constraint on storage technologies, based on their use of `charge_rate`.

.. math::

  energy_{cap}(loc::tech) \leq storage_{cap}(loc::tech) \times charge_{rate}(loc::tech) \times energy_{cap, scale}(loc::tech)


3. ``resource_capacity_constraint_rule``
Add upper and lower bounds for resource_cap.

.. math::

  resource_{cap}(loc::tech) \leq resource_{cap, equals}(loc::tech)

if :math:`resource_{cap, equals}(loc::tech)` exists

else:

.. math::

  resource_{cap}(loc::tech) \leq resource_{cap, max}(loc::tech)

4. ``resource_capacity_equals_energy_capacity_constraint_rule``
Add equality constraint for resource_cap to equal energy_cap, for any technologies which have defined resource_cap_equals_energy_cap.

.. math::

  resource_{cap}(loc::tech) = energy_{cap}(loc::tech)

5. ``resource_area_constraint_rule``
Set upper and lower bounds for resource_area.

.. math::

  resource_{area}(loc::tech) \leq resource_{area, equals}(loc::tech)

if :math:`resource_{cap, equals}(loc::tech)` exists

else:

.. math::

  resource_{area}(loc::tech) \leq resource_{area, max}(loc::tech)

6. ``resource_area_per_energy_capacity_constraint_rule``
Add equality constraint for resource_area to equal a percentage of energy_cap, for any technologies which have defined resource_area_per_energy_cap.

.. math::

  resource_{area}(loc::tech) = energy_{cap}(loc::tech) \times area\_per\_energy\_cap(loc::tech) \forall loc::tech in locs::techs_{area}

7. ``resource_area_capacity_per_loc_constraint_rule``
Set upper bound on use of area for all locations which have `available_area` constraint set. Does not consider resource_area applied to demand technologies.

\sum_{tech} resource_{area}(loc_i::tech) \leq area_{available} \forall i in locs

8. ``energy_capacity_constraint_rule``
Add upper and lower bounds for resource_cap.

.. math::

  energy_{cap}(loc::tech) \leq energy_{cap, equals}(loc::tech) \forall loc::tech in locs::techs

if :math:`energy_{cap, equals}(loc::tech)` exists

else:

.. math::

  energy_{cap}(loc::tech) \leq energy_{cap, max}(loc::tech) \forall loc::tech in locs::techs

9. ``energy_capacity_systemwide_constraint_rule``
Set constraints to limit the capacity of a single technology type across all locations in the model.

.. math::

  \sum_{loc} energy_{cap}(loc::tech_i) = energy_{cap, equals, systemwide}(loc::tech_i) \forall i in techs

if :math:`energy_{cap, equals}(loc::tech)` exists

else:

.. math::

  \sum_{loc} energy_{cap}(loc::tech_i) \leq energy_{cap, max, systemwide}(loc::tech_i) \forall i in techs

10. ``reserve_margin_constraint_rule``
Ensure there is always a percentage additional ``energy_cap``, across all carrier producers in a given location, above the demand for that carrier.

Which limits the resource flow **to** ``supply`` and ``supply_plus`` technologies, or **from** ``demand`` technologies.

For ``supply``:

If the option ``constraints.force_r`` is set to true, then

.. math::

   \frac{c_{prod}(loc::tech::carrier, timestep)}{e_{eff}(loc::tech, timestep)} = r_{avail}(loc::tech, timestep)

If that option is not set:

.. math::

    \frac{c_{prod}(loc::tech::carrier, timestep)}{e_{eff}(loc::tech, timestep)} \leq r_{avail}(loc::tech, timestep)

For ``demand``:

If the option ``constraints.force_r`` is set to true, then

.. math::

   c_{con}(loc::tech::carrier, timestep) \times e_{eff}(loc::tech, timestep) = r_{avail}(loc::tech, timestep)

If that option is not set:

.. math::

  c_{con}(loc::tech::carrier, timestep) \times e_{eff}(loc::tech, timestep) \geq r_{avail}(loc::tech, timestep)

For ``supply_plus``:

If the option ``constraints.force_r`` is set to true, then

.. math::

   r(loc::tech, timestep) = r_{avail}(loc::tech, timestep) \times r_{eff}(loc::tech, timestep)

If that option is not set:

.. math::

  r(loc::tech, timestep) \leq r_{avail}(loc::tech, timestep) \times r_{eff}(loc::tech, timestep)

.. Note:: For all other technology types, defining a resource is irrelevant, so they are not constrained here.

Unit commitment
---------------

Provided by: :func:`calliope.constraints.base.unit_commitment`

Defines constraint ``c_unit_commitment``:

.. math::

   operating\_units(loc::tech, timestep) \leq units(loc::tech)

.. Note:: This constraint only applies to technology-location sets which have ``units.max``, ``units.min``, or ``units.equals`` set in their constraints.

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

   c_{prod}(loc::tech::carrier, timestep) = -1 \times c_{con}(c, y_{remote}, x_{remote}, timestep) \times e_{eff}(loc::tech, timestep) \times e_{eff,perdistance}(loc::tech)

Here, :math:`x_{remote}, y_{remote}` are x and y at the remote end of the transmission technology. For example, for ``(loc::tech) = ('hvdc:region_2', 'region_1')``, the remotes would be ``('hvdc:region_1', 'region_2')``.

:math:`c_{prod}(loc::tech::carrier, timestep)` for ``c='power', y='hvdc:region_2', x='region_1'`` would be the import of power from ``region_2`` to ``region_1``, via a ``hvdc`` connection, at time ``timestep``.

This also shows that transmission technologies can have both a static or time-dependent efficiency (line loss), :math:`energy_{eff}(loc::tech, timestep)`, and a distance-dependent efficiency, :math:`energy_{eff,perdistance}(loc::tech)`.

For more detail on distance-dependent configuration see :doc:`configuration`.

Conversion balance
^^^^^^^^^^^^^^^^^^

The conversion balance is given by

.. math::

   c_{prod}(c_{out}, loc::tech, timestep) = -1 \times c_{con}(c_{in}, loc::tech, timestep) \times e_{eff}(loc::tech, timestep)

The principle is similar to that of the transmission balance. The production of carrier :math:`c_{out}` (the ``carrier_out`` option set for the conversion technology) is driven by the consumption of carrier :math:`c_{in}` (the ``carrier_in`` option set for the conversion technology).

Conversion_plus balance
^^^^^^^^^^^^^^^^^^^^^^^

Conversion plus technologies can have several carriers in and several carriers out, leading to a more complex production/consumption balance.

For the primary carrier(s), the balance is:

.. math::

  \sum\limits_{c_{out_1}} \frac{c_{prod}(c_{out_1}, loc::tech, timestep) }{carrier_{fraction}(c_{out_1})} =  -1 \times \sum\limits_{c_{in_1}} c_{con}(c_{in_1}, loc::tech, timestep) \times carrier_{fraction}(c_{in_1}) \times e_{eff}(x, y, timestep)

Where ``c_{out_1}`` and ``c_{in_1}`` are the sets of primary production and consumption carriers, respectively and ``carrier_{fraction}`` is the relative contribution of these carriers, as defined in ??.

The remaining constraints (``c_balance_conversion_plus_secondary_out``, ``c_balance_conversion_plus_tertiary_out``, ``c_balance_conversion_plus_secondary_in``, ``c_balance_conversion_plus_tertiary_in``) link the input/output of the technology secondary and tertiary carriers to the primary consumption/production.

For production:

.. math::

  \sum\limits_{c_{out_1}} \frac{c_{prod}}{\frac{(c_{out_1}, loc::tech, timestep)}{carrier_{fraction}(c_{out_1})}} \times min(carrier_{fraction}(c_{out_x}))=  \sum\limits_{c_{out_x}} c_{prod}(c_{out_x}, loc::tech, timestep) \times \frac{carrier_{fraction}(c_{out_x})}{min(carrier_{fraction}(c_{out_x}))}

For consumption:

.. math::

  \sum\limits_{c_{in_1}} \frac{c_{con}(c_{in_1}, loc::tech, timestep) }{carrier_{fraction}(c_{in_1})} \times min(carrier_{fraction}(c_{in_x}))=  \sum\limits_{c_{in_x}} c_{con}(c_{in_x}, loc::tech, timestep) \times \frac{carrier_{fraction}(c_{in_x})}{min(carrier_{fraction}(c_{in_x}))}

Where ``x`` is either 2 (secondary carriers) or 3 (tertiary carriers).

.. Warning::

   The ``conversion_plus`` technology is still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior. It is also possible to use a combination of several regular ``conversion`` technologies to achieve some of the behaviors covered by ``conversion_plus``, but at the expense of model complexity.

Supply_plus balance
^^^^^^^^^^^^^^^^^^^

``Supply_plus`` technologies are ``supply`` technologies with more control over resource flow. You can have multiple resources, a resource capacity, and storage of resource before it is converted to the primary carrier_out.

If storage is possible:

.. math::

   s(loc::tech, timestep) = s_{minusone} + r(loc::tech, timestep) + r_{2}(loc::tech, timestep) - c_{prod}

Otherwise:

.. math::

  r(loc::tech, timestep) = c_{prod} - r_{2}


Where:

:math:`c_{prod}` is defined as :math:`\frac{c_{prod}(loc::tech::carrier, timestep)}{total_{eff}}`.

:math:`total_{eff}(loc::tech, timestep)` is defined as :math:`energy_{eff}(loc::tech, timestep) + p_{eff}(loc::tech, timestep)`, the plant efficiency including parasitic losses

:math:`resource_{2}(loc::tech, timestep)` is the secondary resource and is always set to zero unless the technology explicitly defines a secondary resource.

:math:`storage(loc::tech, timestep)` is the storage level at time :math:`t`.

:math:`storage_{minusone}` describes the state of storage at the previous timestep. :math:`storage_{minusone} = s_{init}(loc::tech)` at time :math:`t=0`. Else,

.. math::

   s_{minusone} = (1 - s_{loss}) \times timeres(t-1) \times s(loc::tech, t-1)

.. Note:: In operation mode, ``storage_init`` is carried over from the previous optimization period.


Storage balance
^^^^^^^^^^^^^^^
Storage technologies balance energy charge, energy discharge, and energy stored:

.. math::

   s(loc::tech, timestep) = s_{minusone} - c_{prod} - c_{con}

Where:

:math:`c_{prod}` is defined as :math:`\frac{c_{prod}(loc::tech::carrier, timestep)}{total_{eff}}` if :math:`total_{eff} > 0`, otherwise :math:`c_{prod} = 0`

:math:`c_{con}` is defined as :math:`c_{con}(loc::tech::carrier, timestep) \times total_{eff}`

:math:`total_{eff}(loc::tech, timestep)` is defined as :math:`energy_{eff}(loc::tech, timestep) + p_{eff}(loc::tech, timestep)`, the plant efficiency including parasitic losses

:math:`storage(loc::tech, timestep)` is the storage level at time :math:`t`.

:math:`storage_{minusone}` describes the state of storage at the previous timestep. :math:`storage_{minusone} = s_{init}(loc::tech)` at time :math:`t=0`. Else,

.. math::

   s_{minusone} = (1 - s_{loss}) \times timeres(t-1) \times s(loc::tech, t-1)

.. Note:: In operation mode, ``storage_init`` is carried over from the previous optimization period.


Node build constraints
----------------------

Provided by: :func:`calliope.constraints.base.node_constraints_build`

Built capacity is managed by six constraints.

``c_s_cap``
^^^^^^^^^^^
This constrains the built storage capacity by:

.. math::

    s_{cap}(loc::tech) \leq s_{cap,max}(loc::tech)

If ``y.constraints.s_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

If both :math:`energy_{cap,max}(loc::tech)` and :math:`charge\_rate` are not given, :math:`storage_{cap}(loc::tech)` is automatically set to zero.

If ``y.constraints.s_time.max`` is true at location ``x``, then ``y.constraints.s_time.max`` and ``y.constraints.e_cap.max`` are used to to compute ``storage_cap.max``. The minimum value of ``storage_cap.max`` is taken, based on analysis of all possible time sets which meet the s_time.max value. This allows time-varying efficiency, :math:`energy_{eff}(loc::tech, timestep)` to be accounted for.

If the technology is constrained with integer constraints ``units.max/min/equals`` then the built storage capacity becomes:

.. math::

    s_{cap}(loc::tech) \leq units_{max}(loc::tech) \times s_{cap,per\_unit}

``c_r_cap``
^^^^^^^^^^^
This constrains the built resource conversion capacity by:

.. math::

  r_{cap}(loc::tech) \leq r_{cap,max}(loc::tech)

If the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

``c_r_area``
^^^^^^^^^^^^
This constrains the resource conversion area by:

.. math::

  r_{area}(loc::tech) \leq r_{area,max}(loc::tech)

By default, ``y.constraints.r_area.max`` is set to false, and in that case, :math:`resource_{area}(loc::tech)` is forced to :math:`1.0`. If the model is running in operational mode, the inequality in the equation above is turned into an equality constraint. Finally, if ``y.constraints.r_area_per_e_cap`` is given, then the equation :math:`resource_{area}(loc::tech) = e_{cap}(loc::tech) * r\_area\_per\_cap` applies instead.

``c_e_cap``
^^^^^^^^^^^
This constrains the carrier conversion capacity by:

.. math::
  e_{cap}(loc::tech) \leq e_{cap,max}(loc::tech) \times e\_cap\_scale

If a technology ``y`` is not allowed at a location ``x``, :math:`energy_{cap}(loc::tech) = 0` is forced.

``y.constraints.e_cap_scale`` defaults to 1.0 but can be set on a per-technology, per-location basis if necessary.

If ``y.constraints.e_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

If the technology is constrained with integer constraints ``units.max/min/equals`` then the carrier conversion capacity becomes:

.. math::

    e_{cap}(loc::tech) \leq units_{max}(loc::tech) \times e_{cap,per\_unit}

If the technology is not constrained with integer constraints ``units.max/min/equals``, but does define a ``purchase`` cost then the carrier conversion capacity becomes:

.. math::

    e_{cap}(loc::tech) \leq e_{cap,max}(loc::tech) \times e\_cap\_scale \times purchased(loc::tech)

``c_e_cap_storage``
^^^^^^^^^^^^^^^^^^^
This constrains the carrier conversion capacity for storage technologies by:

.. math::
  e_{cap}(loc::tech) \leq e_{cap,max}

Where :math:`energy_{cap,max} = s_{cap}(loc::tech) * charge\_rate * e\_cap\_scale`

``y.constraints.e_cap_scale`` defaults to 1.0 but can be set on a per-technology, per-location basis if necessary.

If the technology is constrained with integer constraints ``units.max/min/equals`` then the carrier conversion capacity for storage technologies becomes:

.. math::

    e_{cap}(loc::tech) \leq units_{max}(loc::tech) \times e_{cap,per\_unit}

``c_r2_cap``
^^^^^^^^^^^^
This manages the secondary resource conversion capacity by:

.. math::
  r2_{cap}(loc::tech) \leq r2_{cap,max}(loc::tech)

If ``y.constraints.r2_cap.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

There is an additional relevant option, ``y.constraints.r2_cap_follows``, which can be overridden on a per-location basis. It can be set either to ``resource_cap`` or ``energy_cap``, and if set, sets ``c_r2_cap`` to track one of these, ie, :math:`r2_{cap,max} = r_{cap}(loc::tech)` (analogously for ``energy_cap``), and also turns the constraint into an equality constraint.

``c_units``
^^^^^^^^^^^^
This manages the maximum number of integer units by:

.. math::
  units_{cap}(loc::tech) \leq units_{max}(loc::tech)

If ``y.constraints.units.equals`` is set for location ``x`` or the model is running in operational mode, the inequality in the equation above is turned into an equality constraint.

Node operational constraints
----------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_operational`

This component ensures that nodes remain within their operational limits, by constraining ``r``, ``carrier_prod``, ``carrier_con``, ``s``, ``r2``, and ``export``.

``r``
^^^^^
:math:`resource(loc::tech, timestep)` is constrained to remain within :math:`resource_{cap}(loc::tech)`, with the constraint ``c_r_max_upper``:

.. math::

   r(loc::tech, timestep) \leq time\_res(t) \times r_{cap}(loc::tech)

``carrier_prod``
^^^^^^^^^^
:math:`carrier_prod(loc::tech::carrier, timestep)` is constrained by ``carrier_prod_max`` and ``carrier_prod_min``:

.. math::

   c_{prod}(loc::tech::carrier, timestep) \leq time\_res(t) \times e_{cap}(loc::tech) \times p_{eff}(loc::tech, timestep)

if ``c`` is the ``carrier_out`` of ``y``, else :math:`c_{prod}(loc::tech::carrier, y) = 0`.

If ``energy_cap_min_use`` is defined, the minimum output is constrained by:

.. math::

   c_{prod}(loc::tech::carrier, timestep) \geq time\_res(t) \times e_{cap}(loc::tech) \times e_{cap,minuse}

These contraints are skipped for ``conversion_plus`` technologies if ``c`` is not the primary carrier.

If the technology is constrained with integer constraints ``units.max/min/equals`` then `carrier_prod(loc::tech::carrier, timestep)` constraints become:

.. math::

     c_{prod}(loc::tech::carrier, timestep) \leq time\_res(t) \times operating\_units(loc::tech, timestep) \times e_{cap, per\_unit}(loc::tech) \times p_{eff}(loc::tech, timestep)

.. math::

     c_{prod}(loc::tech::carrier, timestep) \geq time\_res(t) \times operating\_units(loc::tech, timestep) \times e_{cap, per\_unit}(loc::tech) \times e_{cap,minuse}


``carrier_con``
^^^^^^^^^
For technologies which are not ``supply`` or ``supply_plus``, :math:`carrier_con(loc::tech::carrier, timestep)` is non-zero. Thus :math:`arrierc_con(loc::tech::carrier, timestep)` is constrainted by ``carrier_con_max``:

.. math::

   c_{con}(loc::tech::carrier, timestep) \geq -1 \times time\_res(t) \times e_{cap}(loc::tech)

and :math:`c_{con}(loc::tech::carrier, timestep) = 0` otherwise.

This constraint is skipped for a ``conversion_plus`` and ``conversion`` technologies If ``c`` is a possible consumption carrier (primary, secondary, or tertiary).

If the technology is constrained with integer constraints ``units.max/min/equals`` then `carrier_con(loc::tech::carrier, timestep)` constraints become:

.. math::

     c_{prod}(loc::tech::carrier, timestep) \geq-1 \times time\_res(t) \times operating\_units(loc::tech, timestep) \times e_{cap, per\_unit}(loc::tech) \times p_{eff}(loc::tech, timestep)

``s``
^^^^^
The constraint ``c_s_max`` ensures that storage cannot exceed its maximum size by

.. math::

   s(loc::tech, timestep) \leq s_{cap}(loc::tech)

``r2``
^^^^^^

``c_r2_max`` constrains the secondary resource by

.. math::

   r2(loc::tech, timestep) \leq timeres(t) \times r2_{cap}(loc::tech)

There is an additional check if ``y.constraints.r2_startup_only`` is true. In this case, :math:`r2(loc::tech, timestep) = 0` unless the current timestep is still within the startup time set in the ``startup_time_bounds`` model-wide setting. This can be useful to prevent undesired edge effects from occurring in the model.

``export``
^^^^^^^^^^

``c_export_max`` constrains the export of a produced carrier by

.. math::

   carrier_export(loc::tech::carrier, timestep) \leq export_{cap}(loc::tech)

If the technology is constrained with integer constraints ``units.max/min/equals`` then `carrier_export(loc::tech::carrier, timestep)` constraint becomes:

.. math::

     carrier_export(loc::tech::carrier, timestep) \leq export_{cap}(loc::tech) \times operating\_units(loc::tech, timestep)

Transmission constraints
------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_transmission`

This component provides a single constraint, ``c_transmission_capacity``, which forces :math:`energy_{cap}` to be symmetric for transmission nodes. For example, for for a given transmission line between :math:`x_1` and :math:`x_2`, using the technology ``hvdc``:

.. math::

   e_{cap}(hvdc:x_2, x_1) = e_{cap}(hvdc:x_1, x_2)

Node costs
----------

Provided by: :func:`calliope.constraints.base.node_costs`

These equations compute costs per node.

Weights are adjusted for individual timesteps depending on the timestep reduction methods applied (see :ref:`run_time_res`), and are given by :math:`W(t)` when computing costs.

The depreciation rate for each cost class ``k`` is calculated as

.. math::

   d(y, cost) = \frac{1}{plant\_life(tech)}

if the interest rate :math:`i` is :math:`0`, else

.. math::

   d(y, cost) = \frac{i \times (1 + i(y, cost))^{plant\_life(k)}}{(1 + i(y, cost))^{plant\_life(k)} - 1}

Costs are split into fixed and variable costs. The total costs are computed in ``c_cost`` by

.. math::

   cost(loc::tech, cost) = cost_{fixed}(loc::tech, cost) + \sum\limits_t cost_{var}(loc::tech, cost, timestep)

The fixed costs include construction costs, annual operation and maintenance (O\&M) costs, and O\&M costs which are a fraction of the construction costs.
The total fixed costs are computed in ``c_cost_fixed`` by

.. math::

  cost_{fixed}(loc::tech, cost) = cost_{con} + cost_{om, frac} \times cost_{con} + cost_{om, fixed} \times e_{cap}(loc::tech) \times \frac{\sum\limits_t timeres(t) \times W(t)}{8760}

Where

.. math::

   cost_{con} &= d(y, cost) \times \frac{\sum\limits_t timeres(t) \times W(t)}{8760} \\
   & \times (cost_{s\_cap}(y, cost) \times s_{cap}(loc::tech) \\
   & + cost_{r\_cap}(y, cost) \times r_{cap}(loc::tech) \\
   & + cost_{r\_area}(y, cost) \times r_{area}(loc::tech) \\
   & + cost_{e\_cap}(y, cost) \times e_{cap}(loc::tech) \\
   & + cost_{r2\_cap}(y, cost) \times r2_{cap}(loc::tech) \\
   & + cost_{purchase}(y, cost) \times units(loc::tech) \\
   & + cost_{purchase}(y, cost) \times purchased(loc::tech))

The costs are as defined in the model definition, e.g. :math:`cost_{r\_cap}(y, cost)` corresponds to ``y.costs.k.r_cap``.

.. Note:: purchase costs occur twice, but will only be applied once, depending on whether the technology constraints trigger an integer decision variable (``units(loc::tech)``) or a binary decision variable (``purchased(loc::tech)``).

For transmission technologies, :math:`cost_{e\_cap}(y, cost)` is computed differently, to include the per-distance costs:

.. math::

   cost_{e\_cap,transmission}(y, cost) = \frac{cost_{e\_cap}(y, cost) + cost_{e\_cap,perdistance}(y, cost)}{2}

This implies that for transmission technologies, the cost of construction is split equally across the two locations connected by the technology.

The variable costs are O&M costs applied at each time step:

.. math::

   cost_{var} = cost_{op,var} + cost_{op,fuel} + cost_{op,r2} + cost_{op, export}

Where:

.. math::
   cost_{op,var} = cost_{om\_var}(k, loc::tech, timestep) \times \sum_t W(t) \times c_{prod}(loc::tech::carrier, timestep)

   cost_{op,fuel} = \frac{cost_{om\_fuel}(k, loc::tech, timestep) \times \sum_t W(t) \times r(loc::tech, timestep)}{r_{eff}(loc::tech)}

   cost_{op,r2} = \frac{cost_{om\_r2}(k, loc::tech, timestep) \times \sum_t W(t) \times r_{2}(loc::tech, timestep)}{r2_{eff}(loc::tech)}

   cost_{op, export} = cost_{export}(k, loc::tech, timestep) \times carrier_export(loc::tech::carrier, timestep)

If :math:`cost_{om\_fuel}(k, loc::tech, timestep)` is given for a ``supply`` technology and :math:`energy_{eff}(loc::tech) > 0` for that technology, then:

.. math::
  cost_{op,fuel} =cost_{om\_fuel}(k, loc::tech, timestep) \times \sum_t W(t) \times \frac{c_{prod}(loc::tech::carrier, timestep)}{e_{eff}(loc::tech)}

``c`` is the technology primary ``carrier_out`` in all cases.


Model balancing constraints
---------------------------

Provided by: :func:`calliope.constraints.base.model_constraints`

Model-wide balancing constraints are constructed for nodes that have children:

.. math::

   \sum_{loc::tech \in X_{i}} c_{prod}(loc::tech::carrier, timestep) + \sum_{loc::tech \in X_{i}} c_{con}(loc::tech::carrier, timestep) = 0 \qquad\forall i, t

:math:`i` are the level 0 locations, and :math:`X_{i}` is the set of level 1 locations (:math:`x`) within the given level 0 location, together with that location itself.

There is also the need to ensure that technologies cannot export more energy than they produce:

.. math::

   c_{prod}(loc::tech::carrier, timestep) \geq carrier_export(loc::tech::carrier, timestep)

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

   \sum_{tech} \sum_{loc} c_{prod}(loc::tech::carrier, t_{max,c}) \times (1 + m_{c}) \leq timeres(t) \times \sum_{y_{c}} \sum_{loc} (e_{cap}(loc::tech) / e_{eff}(loc::tech, t_{max,c}))

where :math:`y_{c}` is the subset of ``y`` that delivers the carrier ``c`` and :math:`m_{c}` is the system margin for that carrier.

For each carrier (with the name ``carrier_name``), Calliope attempts to read the model-wide option ``system_margin.carrier_name``, only applying this constraint if a setting exists.

.. _system_e_cap:

System-wide capacity
--------------------

Provided by: :func:`calliope.constraints.planning.node_constraints_build_total`

This constraint sets a maximum for capacity, ``energy_cap``, across all locations for any given technology:

.. math::

  \sum_{loc} e_{cap}(x, y) \leq e_{cap,total\_max}(tech)

If :math:`energy_{cap,total\_equals}` is given instead, this becomes :math:`\sum_{loc} e_{cap}(x, y) \leq e_{cap,total\_max}(tech)`.

.. math::

   \sum_{tech} \sum_{loc} c_{prod}(loc::tech::carrier, t_{max,c}) \times (1 + m_{c}) \leq timeres(t) \times \sum_{y_{c}} \sum_{loc} (e_{cap}(loc::tech) / e_{eff}(loc::tech, t_{max,c}))

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

   diff = \frac{c_{prod}(loc::tech::carrier, timestep) + c_{con}(loc::tech::carrier, timestep)}{timeres(t)} - \frac{c_{prod}(loc::tech::carrier, t-1) + c_{con}(loc::tech::carrier, t-1)}{timeres(t-1)}

   max\_ramping\_rate = e_{ramping} \times e_{cap}(loc::tech)

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

   \sum_{y^*} \sum_{loc} e_{cap}(loc::tech) \geq fraction \times \sum_{tech} \sum_{loc} e_{cap}(loc::tech)

Here, :math:`y^*` is the subset of :math:`y` given by the specified group, in this example, ``renewables``. :math:`fraction` is the fraction specified, in this example, :math:`0.8`. The relation between the right-hand side and the left-hand side, :math:`\geq`, is determined by the setting given, ``>=``, which can be ``==``, ``<=``, or ``>=``.

If more than one group is listed under ``capacity``, several analogous constraints are set up.

Similarly, ``c_group_fraction_output`` sets up constraints in the form of

.. math::

   \sum_{y^*} \sum_{loc} \sum_t c_{prod}(loc::tech::carrier, timestep) \geq fraction \times \sum_{tech} \sum_{loc} \sum_t c_{prod}(loc::tech::carrier, timestep)

Finally, ``c_group_fraction_demand_power_peak`` sets up constraints in the form of

.. math::

   \sum_{y^*} \sum_{loc} e_{cap}(loc::tech) \geq fraction \times (-1 - m_{c}) \times peak

   peak = \frac{\sum_{loc} r(y_d, x, t_{peak}) \times r_{scale}(y_d, x)}{timeres(t_{peak})}

This assumes the existence of a technology, ``demand_power``, which defines a demand (negative resource). :math:`y_d` is ``demand_power``. :math:`m_{c}` is the capacity margin defined for the carrier ``c`` in the model-wide settings (see :ref:`system_margin`). :math:`t_{peak}` is the timestep where :math:`resource(y_d, x, timestep)` is maximal.

Whether any of these equations are equalities, greater-or-equal-than inequalities, or lesser-or-equal-than inequalities, is determined by whether ``>=``, ``<=``, or ``==`` is given in their respective settings.

Available area
--------------

Provided by: :func:`calliope.constraints.optional.max_r_area_per_loc`

Where several technologies require space to acquire resource (e.g. solar collecting technologies) at a given location, this constraint provides the ability to limit the total area available at a location:

.. math::

  area_{available}(x) \geq \sum_{tech} \sum_{xi} r_{area}(loc::techi)

Where ``xi`` is the set of locations within the family tree, descending from and including ``x``.
