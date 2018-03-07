-----------------
Model formulation
-----------------

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

.. _time_varying_vs_constant_parameters:

Time-varying vs. constant model parameters
------------------------------------------

Some model parameters which are defined over the set of time steps ``timesteps`` can either given as time series or as constant values. If given as constant values, the same value is used for each time step ``timestep``. For details on how to define a parameter as time-varying and how to load time series data into it, see the :ref:`time series description in the model configuration section <configuration_timeseries>`.

Decision variables
------------------

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

Provided by: :func:`calliope.constraints.objective.objective_cost_minimization`

The default objective function minimizes cost:

.. math::

   min: z = \sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{m}))

where :math:`cost_{m}` is the monetary cost class.

Alternative objective functions can be used by setting the ``objective`` in the model configuration (see :ref:`config_reference_model_wide`).

`weight(tech)` is 1 by default, but can be adjusted to change the relative weighting of costs of different technologies in the objective, by setting ``weight`` on any technology (see :ref:`config_reference_techs`).

Basic constraints
-----------------

Energy Balance
^^^^^^^^^^^^^^

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
^^^^^^^^

Constrain the capacity decision variables to maximum/minimum/equals the input parameters given
:func:`calliope.backend.pyomo.constraints.capacity.py`

1. ``storage_capacity_constraint_rule``
Set maximum storage capacity for supply_plus & storage techs only. This can be set by either storage_cap (kWh) or by energy_cap (charge/discharge capacity) * charge rate. If storage_cap_equals and energy_cap_equals are set for the technology, then storage_cap * charge rate = energy_cap must hold. Otherwise, take the lowest capacity defined by storage_cap_max or energy_cap_max / charge rate.

.. math::

  storage_{cap}(loc::tech) \leq storage_{cap, equals}(loc::tech)

if :math:`storage_{cap, equals}(loc::tech)` exists

else:

.. math::

  storage_{cap}(loc::tech) \leq energy_{cap, equals}(loc::tech) \times charge_{rate}

if :math:`energy_{cap, equals}(loc::tech)` and :math:`charge_{rate}(loc::tech)` exist.

else:

.. math::

  storage_cap(loc::tech) \leq storage_{cap, max}(loc::tech)

if :math:`storage_{cap, max}(loc::tech) \leq energy_{cap, max}(loc::tech) \times charge_{rate}`.

else:

.. math::

  storage_{cap}(loc::tech) \leq energy_{cap, max}(loc::tech) \times charge_{rate}

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
