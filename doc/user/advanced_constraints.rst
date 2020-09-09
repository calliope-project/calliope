====================
Advanced constraints
====================

This section, as the title suggests, contains more info and more details, and in particular, information on some of Calliope's more advanced functionality.

We suggest you read the :doc:`building`, :doc:`running` and :doc:`analysing` sections first.

.. _supply_plus:

------------------------
The ``supply_plus`` tech
------------------------

The ``plus`` tech groups offer complex functionality, for technologies which cannot be described easily. ``Supply_plus`` allows a supply technology with internal storage of resource before conversion to the carrier happens. This could be emulated with dummy carriers and a combination of supply, storage, and conversion techs, but the ``supply_plus`` tech allows for concise and mathematically more efficient formulation.

.. figure:: images/supply_plus.*
   :alt: supply_plus

   Representation of the ``supply_plus`` technology

An example use of ``supply_plus`` is to define a concentrating solar power (CSP) technology which consumes a solar resource, has built-in thermal storage, and produces electricity. See the :doc:`national-scale built-in example model <tutorials_01_national>` for an application of this.

See the :ref:`listing of supply_plus configuration <abstract_base_tech_definitions>` in the abstract base tech group definitions for the additional constraints that are possible.

.. Warning:: When analysing results from supply_plus, care must be taken to correctly account for the losses along the transformation from resource to carrier. For example, charging of storage from the resource may have a ``resource_eff``-associated loss with it, while discharging storage to produce the carrier may have a different loss resulting from a combination of ``energy_eff`` and ``parasitic_eff``. Such intermediate conversion losses need to be kept in mind when comparing discharge from storage with ``carrier_prod`` in the same time step.

.. _conversion_plus:

----------------------------
The ``conversion_plus`` tech
----------------------------

The ``plus`` tech groups offer complex functionality, for technologies which cannot be described easily. ``Conversion_plus`` allows several carriers to be converted to several other carriers. Describing such a technology requires that the user understands the ``carrier_ratios``, i.e. the interactions and relative efficiencies of carrier inputs and outputs.

.. figure:: images/conversion_plus.*
   :alt: conversion_plus

   Representation of the most complex ``conversion_plus`` technology available

The ``conversion_plus`` technologies allows for up to three **carrier groups** as inputs (``carrier_in``, ``carrier_in_2`` and ``carrier_in_3``) and up to three carrier groups as outputs (``carrier_out``, ``carrier_out_2`` and ``carrier_out_3``). A carrier group can contain any number of carriers.

The efficiency of a ``conversion_plus`` tech dictates how many units of `carrier_out` are produced per unit of consumed `carrier_in`. A unit of `carrier_out_2` and of `carrier_out_3` is produced each time a unit of `carrier_out` is produced. Similarly, a unit of `Carrier_in_2` and of `carrier_in_3` is consumed each time a unit of `carrier_in` is consumed. Within a given carrier group (e.g. `carrier_out_2`) any number of carriers can meet this one unit. The ``carrier_ratio`` of any carrier compares it either to the production of one unit of `carrier_out` or to the consumption of one unit of `carrier_in`.

In this section, we give examples of a few ``conversion_plus`` technologies alongside the YAML formulation required to construct them:

Combined heat and power
-----------------------

A combined heat and power plant produces electricity, in this case from natural gas. Waste heat that is produced can be used to meet nearby heat demand (e.g. via district heating network). For every unit of electricity produced, 0.8 units of heat are always produced. This is analogous to the heat to power ratio (HTP). Here, the HTP is 0.8.

.. container:: twocol

    .. container:: leftside

        .. figure:: images/conversion_plus_chp.*

    .. container:: rightside

        .. code-block:: yaml

            chp:
                essentials:
                    name: Combined heat and power
                    carrier_in: gas
                    carrier_out: electricity
                    carrier_out_2: heat
                    primary_carrier_out: electricity
                constraints:
                    energy_eff: 0.45
                    energy_cap_max: 100
                    carrier_ratios.carrier_out_2.heat: 0.8


Air source heat pump
--------------------

The output energy from the heat pump can be *either* heat or cooling, simulating a heat pump that can be useful in both summer and winter. For each unit of electricity input, one unit of output is produced. Within this one unit of ``carrier_out``, there can be a combination of heat and cooling. Heat is produced with a COP of 5, cooling with a COP of 3. If only heat were produced in a timestep, 5 units of it would be available in carrier_out; similarly 3 units for cooling. In another timestep, both heat and cooling might be produced with e.g. 2.5 units heat + 1.5 units cooling = 1 unit of carrier_out.

.. figure:: images/conversion_plus_ahp.*

.. code-block:: yaml

    ahp:
        essentials:
            name: Air source heat pump
            carrier_in: electricity
            carrier_out: [heat, cooling]
            primary_carrier_out: heat

        constraints:
            energy_eff: 1
            energy_cap_max: 100
            carrier_ratios:
                carrier_out:
                    heat: 5
                    cooling: 3

Combined cooling, heat and power (CCHP)
---------------------------------------

A CCHP plant can use generated heat to produce cooling via an absorption chiller. As with the CHP plant, electricity is produced at 45% efficiency.  For every unit of electricity produced, 1 unit of ``carrier_out_2`` must be produced, which can be a combination of 0.8 units of heat and 0.5 units of cooling. Some example ways in which the model could decide to operate this unit in a given time step are:

* 1 unit of gas (``carrier_in``) is converted to 0.45 units of electricity (``carrier_out``) and (0.8 * 0.45) units of heat (``carrier_out_2``)
* 1 unit of gas is converted to 0.45 units electricity and (0.5 * 0.45) units of cooling
* 1 unit of gas is converted to 0.45 units electricity, (0.3 * 0.8 * 0.45) units of heat, and (0.7 * 0.5 * 0.45) units of cooling

.. container:: twocol

    .. container:: leftside

        .. figure:: images/conversion_plus_cchp.*

    .. container:: rightside

        .. code-block:: yaml

            cchp:
                essentials:
                    name: Combined cooling, heat and power
                    carrier_in: gas
                    carrier_out: electricity
                    carrier_out_2: [heat, cooling]
                    primary_carrier_out: electricity

                constraints:
                    energy_eff: 0.45
                    energy_cap_max: 100
                    carrier_ratios.carrier_out_2: {heat: 0.8, cooling: 0.5}

Advanced gas turbine
--------------------

This technology can choose to burn methane (CH:sub:`4`) or send hydrogen (H:sub:`2`) through a fuel cell to produce electricity. One unit of carrier_in can be met by any combination of methane and hydrogen. If all methane, 0.5 units of carrier_out would be produced for 1 unit of carrier_in (energy_eff). If all hydrogen, 0.25 units of carrier_out would be produced for the same amount of carrier_in (energy_eff * hydrogen carrier ratio).

.. figure:: images/conversion_plus_gas.*

.. code-block:: yaml

    gt:
        essentials:
            name: Advanced gas turbine
            carrier_in: [methane, hydrogen]
            carrier_out: electricity

        constraints:
            energy_eff: 0.5
            energy_cap_max: 100
            carrier_ratios:
                carrier_in: {methane: 1, hydrogen: 0.5}

Complex fictional technology
----------------------------

There are few instances where using the full capacity of a conversion_plus tech is physically possible. Here, we have a fictional technology that combines fossil fuels with biomass/waste to produce heat, cooling, and electricity. Different 'grades' of heat can be produced, the higher grades having an alternative. High grade heat (``high_T_heat``) is produced and can be used directly, or used to produce electricity (via e.g. organic rankine cycle). ``carrier_out`` is thus a combination of these two. `carrier_out_2` can be 0.3 units mid grade heat for every unit `carrier_out` or 0.2 units cooling. Finally, 0.1 units ``carrier_out_3``, low grade heat, is produced for every unit of `carrier_out`.

.. container:: twocol

    .. container:: leftside

        .. figure:: images/conversion_plus_complex.*

    .. container:: rightside

        .. code-block:: yaml

            complex:
                essentials:
                    name: Complex fictional technology
                    carrier_in: [coal, gas, oil]
                    carrier_in_2: [biomass, waste]
                    carrier_out: [high_T_heat, electricity]
                    carrier_out_2: [mid_T_heat, cooling]
                    carrier_out_3: low_T_heat
                    primary_carrier_out: electricity

                constraints:
                    energy_eff: 1
                    energy_cap_max: 100
                    carrier_ratios:
                        carrier_in: {coal: 1.2, gas: 1, oil: 1.6}
                        carrier_in_2: {biomass: 1, waste: 1.25}
                        carrier_out: {high_T_heat: 0.8, electricity: 0.6}
                        carrier_out_2: {mid_T_heat: 0.3, cooling: 0.2}
                        carrier_out_3.low_T_heat: 0.15

A ``primary_carrier_out`` must be defined when there are multiple ``carrier_out`` values defined, similarly ``primary_carrier_in`` can be defined for ``carrier_in``. `primary_carriers` can be defined as any carrier in a technology's input/output carriers (including secondary and tertiary carriers). The chosen output carrier will be the one to which production costs are applied (reciprocally, input carrier for consumption costs).

.. note:: ``Conversion_plus`` technologies can also export any one of their output carriers, by specifying that carrier as ``carrier_export``.

-------------------------
Resource area constraints
-------------------------

Several optional constraints can be used to specify area-related restrictions on technology use.

To make use of these constraints, one should set ``resource_unit: energy_per_area`` for the given technologies. This scales the available resource at a given location for a given technology with its ``resource_area`` decision variable.

The following related settings are available:

* ``resource_area_equals``, ``resource_area_max``, ``resource_area_min``: Set uppper or lower bounds on resource_area or force it to a specific value
* ``resource_area_per_energy_cap``: False by default, but if set to true, it forces ``resource_area`` to follow ``energy_cap`` with the given numerical ratio (e.g. setting to 1.5 means that ``resource_area == 1.5 * energy_cap``)

By default, ``resource_area_max`` is infinite and ``resource_area_min`` is 0 (zero).

.. _group_constraints:

-----------------
Group constraints
-----------------

Group constraints are applied to named sets of locations and techs, called "constraint groups", specified through a top-level ``group_constraints`` key (sitting alongside other top-level keys like ``model`` and ``run``).

The below example shows two such named groups. The first does not specify a subset of techs or locations and is thus applied across the entire model. In the example, we use ``cost_max`` with the ``co2`` cost class to specify a model-wide emissions limit (assuming the technologies in the model have ``co2`` costs associated with them). We also use the ``demand_share_min`` constraint to force wind and PV to supply at least 40% of electricity demand in Germany, which is modelled as two locations (North and South):

.. code-block:: yaml

    run:
        ...

    model:
        ...

    group_constraints:
        # A constraint group to apply a systemwide CO2 cap
        systemwide_co2_cap:
            cost_max:
                co2: 100000
        # A constraint group to enforce renewable generation in Germany
        renewable_minimum_share_in_germany:
            techs: ['wind', 'pv']
            locs: ['germany_north', 'germany_south']
            demand_share_min:
                electricity: 0.4

When specifying group constraints, a named group must give at least one constraint, but can list an arbitrary amount of constraints, and optionally give a subset of techs and locations:

.. code-block:: yaml

    group_constraints:
        group_name:
            techs: []  # Optional, can be left out if empty
            locs: []  # Optional, can be left out if empty
            # Any number of constraints can be specified for the given group
            constraint_1: ...
            constraint_2: ...
            ...

The below table lists all available group constraints.

Note that when computing the share for ``demand_share`` constraints, only ``demand`` technologies are counted, and that when computing the share for ``supply_share`` constraints, ``supply`` and ``supply_plus`` technologies are counted.

.. list-table:: Group constraints
   :widths: 15 15 60
   :header-rows: 1

   * - Constraint
     - Dimensions
     - Description
   * - ``demand_share_min``
     - carriers
     - Minimum share of carrier demand met from a set of technologies across a set of locations, on average over the entire model period.
   * - ``demand_share_max``
     - carriers
     - Maximum share of carrier demand met from a set of technologies across a set of locations, on average over the entire model period.
   * - ``demand_share_equals``
     - carriers
     - Share of carrier demand met from a set of technologies across a set of locations, on average over the entire model period.
   * - ``demand_share_per_timestep_min``
     - carriers
     - Minimum share of carrier demand met from a set of technologies across a set of locations, in each individual timestep.
   * - ``demand_share_per_timestep_max``
     - carriers
     - Maximum share of carrier demand met from a set of technologies across a set of locations, in each individual timestep.
   * - ``demand_share_per_timestep_equals``
     - carriers
     - Share of carrier demand met from a set of technologies across a set of locations, in each individual timestep.
   * - ``demand_share_per_timestep_decision``
     - carriers
     - Turns the per-timestep share of carrier demand met from a set of technologies across a set of locations into a model decision variable.
   * - ``carrier_prod_share_min``
     - carriers
     - Minimum share of carrier production met from a set of technologies across a set of locations, on average over the entire model period.
   * - ``carrier_prod_share_max``
     - carriers
     - Maximum share of carrier production met from a set of technologies across a set of locations, on average over the entire model period.
   * - ``carrier_prod_share_equals``
     - carriers
     - Share of carrier production met from a set of technologies across a set of locations, on average over the entire model period.
   * - ``carrier_prod_share_per_timestep_min``
     - carriers
     - Minimum share of carrier production met from a set of technologies across a set of locations, in each individual timestep.
   * - ``carrier_prod_share_per_timestep_max``
     - carriers
     - Maximum share of carrier production met from a set of technologies across a set of locations, in each individual timestep.
   * - ``carrier_prod_share_per_timestep_equals``
     - carriers
     - Share of carrier production met from a set of technologies across a set of locations, in each individual timestep.
   * - ``net_import_share_min``
     - carriers
     - Minimum share of demand met from transmission technologies into a set of locations, on average over the entire model period. All transmission technologies of the chosen carrier are added automatically and technologies must thus not be defined explicitly.
   * - ``net_import_share_max``
     - carriers
     - Maximum share of demand met from transmission technologies into a set of locations, on average over the entire model period. All transmission technologies of the chosen carrier are added automatically and technologies must thus not be defined explicitly.
   * - ``net_import_share_equals``
     - carriers
     - Share of demand met from transmission technologies into a set of locations, on average over the entire model. All transmission technologies of the chosen carrier are added automatically and technologies must thus not be defined explicitly. period.
   * - ``carrier_prod_min``
     - carriers
     - Maximum absolute sum of supplied energy (`carrier_prod`) over all timesteps for a set of technologies across a set of locations.
   * - ``carrier_prod_max``
     - carriers
     - Maximum absolute sum of supplied energy (`carrier_prod`) over all timesteps for a set of technologies across a set of locations.
   * - ``carrier_prod_equals``
     - carriers
     - Exact absolute sum of supplied energy (`carrier_prod`) over all timesteps for a set of technologies across a set of locations.
   * - ``cost_max``
     - costs
     - Maximum total cost from a set of technologies across a set of locations.
   * - ``cost_min``
     - costs
     - Minimum total cost from a set of technologies across a set of locations.
   * - ``cost_equals``
     - costs
     - Total cost from a set of technologies across a set of locations must equal given value.
   * - ``cost_var_max``
     - costs
     - Maximum variable cost from a set of technologies across a set of locations.
   * - ``cost_var_min``
     - costs
     - Minimum variable cost from a set of technologies across a set of locations.
   * - ``cost_var_equals``
     - costs
     - Variable cost from a set of technologies across a set of locations must equal given value.
   * - ``cost_investment_max``
     - costs
     - Maximum investment cost from a set of technologies across a set of locations.
   * - ``cost_investment_min``
     - costs
     - Minimum investment cost from a set of technologies across a set of locations.
   * - ``cost_investment_equals``
     - costs
     - Investment cost from a set of technologies across a set of locations must equal given value.
   * - ``energy_cap_share_min``
     - –
     - Minimum share of installed capacity from a set of technologies across a set of locations.
   * - ``energy_cap_share_max``
     - –
     - Maximum share of installed capacity from a set of technologies across a set of locations.
   * - ``energy_cap_share_equals``
     - –
     - Exact share of installed capacity from a set of technologies across a set of locations.
   * - ``energy_cap_min``
     - –
     - Minimum installed capacity from a set of technologies across a set of locations.
   * - ``energy_cap_max``
     - –
     - Maximum installed capacity from a set of technologies across a set of locations.
   * - ``energy_cap_equals``
     - –
     - Exact installed capacity from a set of technologies across a set of locations.
   * - ``resource_area_min``
     - –
     - Minimum resource area used by a set of technologies across a set of locations.
   * - ``resource_area_max``
     - –
     - Maximum resource area used by a set of technologies across a set of locations.
   * - ``resource_area_equals``
     - –
     - Exact resource area used by a set of technologies across a set of locations.


For specifics of the mathematical formulation of the available group constraints, see :ref:`constraint_group` in the mathematical formulation section.

.. seealso:: The :ref:`built-in national-scale example <examplemodels_nationalscale_settings>`'s ``scenarios.yaml`` shows two example uses of group constraints: limiting shared capacity with ``energy_cap_max`` and enforcing a minimum shared power generation with ``carrier_prod_share_min``.


``demand_share_per_timestep_decision``
--------------------------------------

The ``demand_share_per_timestep_decision`` constraint is a special case amongst group constraints, as it introduces a new decision variable, allowing the model to set the share of demand met by each technology given in the constraint's group, across the locations given in the group. The fraction set in the constraint is the fraction of total demand over which the model has control. Setting this to anything else than ``1.0`` only makes sense when a subset of technologies is targeted by the constraint.

It can also be set to ``.inf`` to permit Calliope to decide on the fraction of total demand to cover by the constraint. This can be necessary in cases where there are sources of carrier consumption other than demand in the locations covered by the group constraint: when using conversion techs or when there are losses from storage and transmission, as the share may then be higher than 1, leading to an infeasible model if it is forced to ``1.0``.

This constraint can be useful in large-scale models where individual technologies should not fluctuate in their relative share from time step to time step, for example, when modelling the relative share of heating demand from different heating technologies.

.. Warning:: It is easy to create an infeasible model by setting several conflicting group constraints, in particular when ``demand_share_per_timestep_decision`` is involved. Make sure you think through the implications when setting up these constraints!

----------------------------------
Per-distance constraints and costs
----------------------------------

Transmission technologies can additionally specify per-distance efficiency (loss) with ``energy_eff_per_distance`` and per-distance costs with ``energy_cap_per_distance``:

.. code-block:: yaml

    techs:
        my_transmission_tech:
            essentials:
                ...
            constraints:
                # "efficiency" (1-loss) per unit of distance
                energy_eff_per_distance: 0.99
            costs:
                monetary:
                    # cost per unit of distance
                    energy_cap_per_distance: 10

The distance is specified in transmission links:

.. code-block:: yaml

    links:
        location1,location2:
            my_transmission_tech:
                distance: 500
                constraints:
                    energy_cap.max: 10000

If no distance is given, but the locations have been given lat and lon coordinates, Calliope will compute distances automatically (based on the length of a straight line connecting the locations).

--------------------------
One-way transmission links
--------------------------

Transmission links are bidirectional by default. To force unidirectionality for a given technology along a given link, you have to set the ``one_way`` constraint in the constraint definition of that technology, for that link:

.. code-block:: yaml

    links:
        location1,location2:
            transmission-tech:
                constraints:
                    one_way: true

This will only allow transmission from ``location1`` to ``location2``. To swap the direction, the link name must be inverted, i.e. ``location2,location1``.

--------------
Cyclic storage
--------------

With ``storage`` and ``supply_plus`` techs, it is possible to link the storage at either end of the timeseries, using cyclic storage. This allows the user to better represent multiple years by just modelling one year. Cyclic storage is activated by default (to deactivate: ``run.cyclic_storage: false``). As a result, a technology's initial stored energy at a given location will be equal to its stored energy at the end of the model's last timestep.

For example, for a model running over a full year at hourly resolution, the initial storage at `Jan 1st 00:00:00` will be forced equal to the storage at the end of the timestep `Dec 31st 23:00:00`. By setting ``storage_initial`` for a technology, it is also possible to fix the value in the last timestep. For instance, with ``run.cyclic_storage: true`` and a ``storage_initial`` of zero, the stored energy *must* be zero by the end of the time horizon.

Without cyclic storage in place (as was the case prior to v0.6.2), the storage tech can have any amount of stored energy by the end of the timeseries. This may prove useful in some cases, but has less physical meaning than assuming cyclic storage.

.. note:: Cyclic storage also functions when time clustering, if allowing storage to be tracked between clusters (see :ref:`time_clustering`). However, it cannot be used in ``operate`` run mode.

------------------
Revenue and export
------------------

It is possible to specify revenues for technologies simply by setting a negative cost value. For example, to consider a feed-in tariff for PV generation, it could be given a negative operational cost equal to the real operational cost minus the level of feed-in tariff received.

Export is an extension of this, allowing an energy carrier to be removed from the system without meeting demand. This is analogous to e.g. domestic PV technologies being able to export excess electricity to the national grid. A cost (or negative cost: revenue) can then be applied to export.

.. note:: Negative costs can be applied to capacity costs, but the user must an ensure a capacity limit has been set. Otherwise, optimisation will be unbounded.

.. _group_share:

-------------------------------------------
The ``group_share`` constraint (deprecated)
-------------------------------------------

.. Warning:: ``group_share`` is deprecated as of v0.6.4 and will be removed in v0.7.0. Use the new, more flexible functionality :ref:`group_constraints` to replace it.

The ``group_share`` constraint can be used to force groups of technologies to fulfill certain shares of supply or capacity.

For example, assuming a model containing a ``csp`` and a ``cold_fusion`` power generation technology, we could force at least 85% of power generation in the model to come from these two technologies with the following constraint definition in the ``model`` settings:

.. code-block:: yaml

    model:
        group_share:
            csp,cold_fusion:
                carrier_prod_min:
                    power: 0.85

Possible ``group_share`` constraints with carrier-specific settings are:

* ``carrier_prod_min``
* ``carrier_prod_max``
* ``carrier_prod_equals``

Possible ``group_share`` constraints with carrier-independent settings are:

* ``energy_cap_min``
* ``energy_cap_max``
* ``energy_cap_equals``

These can be implemented as, for example, to force at most 20% of ``energy_cap`` to come from the two listed technologies:

.. code-block:: yaml

    model:
        group_share:
            csp,cold_fusion:
                energy_cap_max: 0.20

------------------------------------
Binary and mixed-integer constraints
------------------------------------

Calliope models are purely linear by default. However, several constraints can turn a model into a binary or mixed-integer model. Because solving problems with binary or integer variables takes considerably longer than solving purely linear models, it usually makes sense to carefully consider whether the research question really necessitates going beyond a purely linear model.

By applying a ``purchase`` cost to a technology, that technology will have a binary variable associated with it, describing whether or not it has been "purchased".

By applying ``units.max``, ``units.min``, or ``units.equals`` to a technology, that technology will have a integer variable associated with it, describing how many of that technology have been "purchased". If a ``purchase`` cost has been applied to this same technology, the purchasing cost will be applied per unit.

.. Warning::

   Integer and binary variables are a recent addition to Calliope and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

.. seealso:: :ref:`milp_example_model`

Asynchronous energy production/consumption
------------------------------------------

The ``asynchronous_prod_con`` binary constraint ensures that only one of ``carrier_prod`` and ``carrier_con`` can be non-zero in a given timestep.

This constraint can be applied to storage or transmission technologies. This example shows use with a heat transmission technology:

.. literalinclude:: ../../calliope/example_models/urban_scale/scenarios.yaml
   :language: yaml
   :dedent: 8
   :start-after: # heat_pipes-start
   :end-before: # heat_pipes-end

In the above example, heat pipes which distribute thermal energy in the network may be prone to dissipating heat in an unphysical way. I.e. given that they have distribution losses associated with them, in any given timestep, a link could produce and consume energy in the same timestep, losing energy to the atmosphere in both instances, but having a net energy transmission of zero. This might allow e.g. a CHP facility to overproduce heat to produce more cheap electricity, and have some way of dumping that heat. Enabling the ``asynchronous_prod_con`` constraint ensures that this does not happen.

-------------------------------
User-defined custom constraints
-------------------------------

It is possible to pass custom constraints to the Pyomo backend, using the :ref:`backend interface <api_backend_interface>`. This requires an understanding of the structure of Pyomo constraints. As an example, the following code reproduces the constraint which limits the maximum carrier consumption to less than or equal to the technology capacity:

.. code-block:: python

    model = calliope.Model(...)
    model.run()  # or `model.run(build_only=True)` if you don't want the model to be optimised before adding the new constraint

    constraint_name = 'max_capacity_90_constraint'
    constraint_sets = ['loc_techs_supply']

    def max_capacity_90_constraint_rule(backend_model, loc_tech):

        return backend_model.energy_cap[loc_tech] <= (
            backend_model.energy_cap_max[loc_tech] * 0.9
        )

    # Add the constraint
    model.backend.add_constraint(constraint_name, constraint_sets, max_capacity_90_constraint_rule)

    # Rerun the model with new constraint.
    new_model = model.backend.rerun()  # `new_model` is a calliope model *without* a backend, it is only useful for saving the results to file

.. note::
    * We like the convention that constraint names end with 'constraint' and constraint rules have the same text, with an appended '_rule', but you are not required to follow this convention to have a working constraint.
    * :python:`model.run(force_rerun=True)` will *not* implement the new constraint, :python:`model.backend.rerun()` is required. If you run :python:`model.run(force_rerun=True)`, the backend model will be rebuilt, killing any changes you've made.