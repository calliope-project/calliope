---------------------------------------------
Model components and mathematical formulation
---------------------------------------------

This section details the mathematical formulation of the different components. For each component, a link to the actual implementing function in the Calliope code is given.

Terminology
-----------

The terminology defined here is used throughout the documentation and the model code and configuration files:

* **Technology**: a technology that produces, consumes, converts or transports energy
* **Location**: a site which can contain multiple technologies and which may contain other locations for energy balancing purposes
* **Node**: a combination of technology and location resulting in specific energy balance equations (:ref:`see below <node_energy_balance>`)
* **Resource**: a source or sink of energy that can (or must) be used by a technology to introduce into or remove energy from the system
* **Carrier**: an energy carrier that groups technologies together into the same network, for example ``electricity`` or ``heat``.

As more generally in constrained optimization, the following terms are also used:

* Parameter: a fixed coefficient that enters into model equations
* Variable: a variable coefficient (decision variable) that enters into model equations
* Set: an index in the algebraic formulation of the equations
* Constraint: an equality or inequality expression that constrains one or several variables

Index sets
----------

Most parameters, variables, and constraints are formulated with respect to at least some of the indices below:

* ``carriers``: carriers
* ``techs``: technologies
* ``locs``: locations
* ``timesteps``: time steps
* ``costs``: cost classes

In some cases, these index sets may have only a single member. For example, if only the power system is modeled, the set ``carriers`` will have a single member, ``power``.

When processed, these sets are often concatenated to avoid sparse matrices. For instance, if a technology ``boiler`` only exists in location ``X1`` and not in locations ``X2`` or ``X3``, then we will specify parameters for just the ``loc::tech`` ``X1::boiler``. This can be extended to parameters which also consider ``carriers``, such that we would have a ``loc::tech::carrier`` ``X1::boiler::heat`` (avoiding empty parameter values for ``power``, as the boiler never considers that enery carrier).

.. _technology_types:

Technology types
----------------

Each technology (that is, each member of the set ``techs``) is of a specific *technology type*, which determines how the framework models the technology and what properties it can have. The technology type is specified by inheritance from one of seven abstract base technologies (see :ref:`configuration_techs` in the model configuration section for more details on this inheritance model):

* Supply: Supplies energy from a resource to a carrier (a source) (base technology: ``supply``)
* Supply_plus: A more feature rich version of ``supply``. It can have storage of resource before conversion to carrier, can define an additional secondary resource, and can have several more intermediate loss factors (base technology: ``supply_plus``)
* Demand: Acts like supply but with a resource that is negative (a sink). Draws energy from a carrier to satisfy a resource demand (base technology: ``demand``)
* Conversion: Converts energy from one carrier to another, can have neither resource nor storage associated with it (base technology: ``conversion``)
* Conversion_plus: A more feature rich version of ``conversion``. There can be several carriers in, converted to several carriers out (base technology: ``conversion_plus``)
* Storage: Can store energy of a specific carrier, cannot have any resource (base technology: ``storage``)
* Transmission: Transports energy of a specific carrier from one location to another, can have neither resource nor storage (base technology: ``transmission``)

The internal definition of these abstract base technologies is given in the :ref:`model configuration reference <abstract_base_tech_definitions>`.

Cost classes
------------

Costs are modeled in Calliope via *cost classes*. By default, only one classes is defined: ``monetary``.

Technologies can define costs for components (installed capacity), for operation & maintenance, and for export for any cost class. Costs can be given as negative values, which defines a revenue rather than a cost.

The primary cost class, ``monetary``, is used to calculate levelized costs and by default enters into the objective function. Therefore each technology should define at least one cost parameter, as it would be considered free otherwise. By default, any cost not specified is assumed to be zero.

Only the ``monetary`` cost class is entered into the default objective function, but other cost classes can be defined for accounting purposes, e.g. ``emissions`` to account for greenhouse gas emissions. Additional cost classes can be created simply by adding them to the definition of costs for a technology (see the :doc:`model configuration section <ref_model_config>` for more detail on this).

To add additional cost classes to the objective function (e.g. ``emissions``), a custom objective function would need to be created. See :ref:`config_reference_model_wide` in model configuration for more details.

Revenue
-------

It is possible to specify revenues for technologies simply by setting a negative cost value. For example, to consider a feed-in tariff for PV generation, it could be given a negative operational cost equal to the real operational cost minus the level of feed-in tariff received.

Putting technologies and locations together: Nodes
--------------------------------------------------

In the model definition, locations can be defined, and for each location (or for groups of locations), technologies can be permitted. The details of this are laid out in the :doc:`model configuration section <ref_model_config>`.

A *node* is the combination of a specific location and technology, and is how Calliope internally builds the model. For a given location, ``loc``, and technology, ``tech``, a set of equations defined over ``loc::tech`` models that specific node.

The most important node variables are laid out below, but more detail is also available in the section :doc:`ref_formulation`.

.. _node_energy_balance:

Node energy balance
-------------------

The basic formulation of each node uses a set of energy balance equations. Depending on the technology type, different energy balance variables are used:

* ``storage(loc::tech, timestep)``: storage level at time ``timestep``
    This is used for ``storage`` and ``supply_plus`` technologies.
* ``resource(loc::tech, timestep)``: resource to technology (+ production) at time ``timestep``. If storage is defined for ``supply_plus``, this is resource to storage flow.
    This is used for ``supply_plus`` technologies.
* ``carrier_prod(loc::tech::carrier, timestep)``: production of a given energy carrier by a technology (+ supply) at time ``timestep``.
    This is used for all technologies, except ``demand``.
* ``c_con(loc::tech::carrier, timestep)``: consumption of a given energy carrier by a technology at time ``timestep``
    This is used for all technologies, except ``supply`` and ``supply_plus``.

The resulting losses associated with energy balancing also depend on the technology type. Each technology node is mapped here, with details on interactions given in :doc:`ref_model_config`.

.. figure:: images/nodes.*
   :alt: Layout of a various node and their energy balance

   The layout of nodes, and their energy balance variables, associated with each technology type. The outward arrows show where losses occur. Depending on a technology, some of these steps may be skipped. For example, most ``supply_plus`` technologies will have no parasitic losses.

Each node can also have the following capacity variables:

* ``storage_cap(loc::tech)``: installed storage capacity
    This is used for ``storage`` and ``supply_plus`` technologies.
* ``resource_cap(loc::tech)``: installed resource to storage conversion capacity
    This is used for ``supply_plus`` technologies.
* ``resource_area(loc::tech)``: installed resource collector area
    This is used for ``supply``, ``supply_plus``, and ``demand`` technologies.
* ``energy_cap(loc::tech)``: installed storage to carrier conversion capacity
    This is used for all technologies.

.. Note:: For nodes that have an internal (parasitic) energy consumption, ``energy_cap_net`` is also included in the solution. This specifies the net conversion capacity, while ``energy_cap`` is gross capacity.

When defining a technology, it must be given at least some constraints, that is, options that describe the functioning of the technology. If not specified, all of these are inherited from the default technology definition (with default values being ``0`` for capacities and ``1`` for efficiencies). Some examples of such options are:

* ``resource(loc::tech, timestep)``: available resource (+ source, - sink)
* ``storage_cap_max(loc::tech)``: maximum storage capacity
* ``storage_loss(loc::tech, timestep)``: storage loss rate
* ``resource_area_max(loc::tech)``: maximum resource collector area
* ``resource_eff(loc::tech)``: resource efficiency
* ``resource_cap_max(loc::tech)``: maximum resource to storage conversion capacity
* ``energy_eff(loc::tech, timestep)``: resource/storage/carrier_in to carrier_out conversion efficiency
* ``energy_cap_max(loc::tech)``: maximum installed carrier conversion capacity, applied to carrier_out

.. Note:: Generally, these constraints are defined on a per-technology basis. However, some (but not all) of them may be overridden on a per-location basis. This allows, for example, setting different constraints on the allowed maximum capacity for a specific technology at each location separately. See :doc:`ref_model_config` for details on this. Once processed in Calliope, all constraints will be indexed over location::technology sets.

Finally, each node tracks its costs (+ costs, - revenue), formulated in two constraints (more details in the :doc:`ref_formulation` section):

* ``cost_investment``: static investment costs, for construction and fixed operational and maintenance (O&M) (i.e., costs per unit of installed capacity)
* ``cost_var``: variable O&M and export costs (i.e., costs per produced unit of output)

.. Note:: Efficiencies, available resources, and costs can be defined to vary in time. Equally (and more likely) they can be given as single values. For more detail on time-varying versus constant values, see :ref:`the corresponding section <time_varying_vs_constant_parameters>` in the model formulation chapter.

Locations and links
-------------------

.. figure:: images/nodes_network.*
   :alt: Layout of linked locations

   Schematic of location linking, including interaction of resource, nodes, and energy carriers. The dashed box defines the system under consideration. Resource flows (green) are lossless, whereas losses can occur along transmission links (black).

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
