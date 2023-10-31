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

.. Warning:: When analysing results from supply_plus, care must be taken to correctly account for the losses along the transformation from resource to carrier. For example, charging of storage from the resource may have a ``source_eff``-associated loss with it, while discharging storage to produce the carrier may have a different loss resulting from a combination of ``flow_out_eff`` and ``parasitic_eff``. Such intermediate conversion losses need to be kept in mind when comparing discharge from storage with ``flow_out`` in the same time step.

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
                    flow_out_eff: 0.45
                    flow_cap_max: 100
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
            flow_out_eff: 1
            flow_cap_max: 100
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
                    flow_out_eff: 0.45
                    flow_cap_max: 100
                    carrier_ratios.carrier_out_2: {heat: 0.8, cooling: 0.5}

Advanced gas turbine
--------------------

This technology can choose to burn methane (CH:sub:`4`) or send hydrogen (H:sub:`2`) through a fuel cell to produce electricity. One unit of carrier_in can be met by any combination of methane and hydrogen. If all methane, 0.5 units of carrier_out would be produced for 1 unit of carrier_in (`flow_out_eff`). If all hydrogen, 0.25 units of carrier_out would be produced for the same amount of carrier_in (`flow_out_eff` * hydrogen carrier ratio).

.. figure:: images/conversion_plus_gas.*

.. code-block:: yaml

    gt:
        essentials:
            name: Advanced gas turbine
            carrier_in: [methane, hydrogen]
            carrier_out: electricity

        constraints:
            flow_out_eff: 0.5
            flow_cap_max: 100
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
                    flow_out_eff: 1
                    flow_cap_max: 100
                    carrier_ratios:
                        carrier_in: {coal: 1.2, gas: 1, oil: 1.6}
                        carrier_in_2: {biomass: 1, waste: 1.25}
                        carrier_out: {high_T_heat: 0.8, electricity: 0.6}
                        carrier_out_2: {mid_T_heat: 0.3, cooling: 0.2}
                        carrier_out_3.low_T_heat: 0.15

A ``primary_carrier_out`` must be defined when there are multiple ``carrier_out`` values defined, similarly ``primary_carrier_in`` can be defined for ``carrier_in``. `primary_carriers` can be defined as any carrier in a technology's input/output carriers (including secondary and tertiary carriers). The chosen output carrier will be the one to which production costs are applied (reciprocally, input carrier for consumption costs).

.. note:: ``Conversion_plus`` technologies can also export any one of their output carriers, by specifying that carrier as the ``export_carrier``.

--------------------
Area use constraints
--------------------

Several optional constraints can be used to specify area-related restrictions on technology use.

To make use of these constraints, one should set ``source_unit: per_area`` for the given technologies. This scales the available source at a given location for a given technology with its ``area_use`` decision variable.

The following related settings are available:

* ``area_use_max``, ``area_use_min``: Set upper or lower bounds on area_use
* ``area_use_per_flow_cap``: False by default, but if set to true, it forces ``area_use`` to follow ``flow_cap`` with the given numerical ratio (e.g. setting to 1.5 means that ``area_use == 1.5 * flow_cap``)

By default, ``area_use_max`` is infinite and ``area_use_min`` is 0 (zero).

----------------------------------
Per-distance constraints and costs
----------------------------------

Transmission technologies can additionally specify per-distance efficiency (loss) with ``flow_out_eff_per_distance`` and per-distance costs with ``flow_cap_per_distance``:

.. code-block:: yaml

    techs:
        my_transmission_tech:
            essentials:
                ...
            constraints:
                # "efficiency" (1-loss) per unit of distance
                flow_out_eff_per_distance: 0.99
            costs:
                monetary:
                    # cost per unit of distance
                    flow_cap_per_distance: 10

The distance is specified in transmission links:

.. code-block:: yaml

    links:
        location1,location2:
            my_transmission_tech:
                distance: 500
                constraints:
                    flow_cap.max: 10000

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

With ``storage`` and ``supply_plus`` techs, it is possible to link the storage at either end of the timeseries, using cyclic storage. This allows the user to better represent multiple years by just modelling one year. Cyclic storage is activated by default (to deactivate: ``config.build.cyclic_storage: false``). As a result, a technology's initial stored energy at a given location will be equal to its stored energy at the end of the model's last timestep.

For example, for a model running over a full year at hourly resolution, the initial storage at `Jan 1st 00:00:00` will be forced equal to the storage at the end of the timestep `Dec 31st 23:00:00`. By setting ``storage_initial`` for a technology, it is also possible to fix the value in the last timestep. For instance, with ``config.build.cyclic_storage: true`` and a ``storage_initial`` of zero, the stored energy *must* be zero by the end of the time horizon.

Without cyclic storage in place (as was the case prior to v0.6.2), the storage tech can have any amount of stored energy by the end of the timeseries. This may prove useful in some cases, but has less physical meaning than assuming cyclic storage.

.. note:: Cyclic storage also functions when time clustering, if allowing storage to be tracked between clusters (see :ref:`time_clustering`). However, it cannot be used in ``operate`` run mode.

------------------
Revenue and export
------------------

It is possible to specify revenues for technologies simply by setting a negative cost value. For example, to consider a feed-in tariff for PV generation, it could be given a negative operational cost equal to the real operational cost minus the level of feed-in tariff received.

Export is an extension of this, allowing a carrier to be removed from the system without meeting demand. This is analogous to e.g. domestic PV technologies being able to export excess electricity to the national grid. A cost (or negative cost: revenue) can then be applied to export.

.. note:: Negative costs can be applied to capacity costs, but the user must an ensure a capacity limit has been set. Otherwise, optimisation will be unbounded.

------------------------------------
Binary and mixed-integer constraints
------------------------------------

Calliope models are purely linear by default. However, several constraints can turn a model into a binary or mixed-integer model. Because solving problems with binary or integer variables takes considerably longer than solving purely linear models, it usually makes sense to carefully consider whether the research question really necessitates going beyond a purely linear model.

By applying a ``purchase`` cost to a technology, that technology will have a binary variable associated with it, describing whether or not it has been "purchased".

By applying ``units_max`` or ``units_min`` to a technology, that technology will have a integer variable associated with it, describing how many of that technology have been "purchased". If a ``purchase`` cost has been applied to this same technology, the purchasing cost will be applied per unit.

.. Warning::

   Integer and binary variables are a recent addition to Calliope and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

.. seealso:: :ref:`milp_example_model`

Asynchronous flow in/out
------------------------

The ``async_flow_switch`` binary variable ensures that only one of ``flow_out`` and ``flow_in`` can be non-zero in a given timestep.

This constraint can be applied to storage or transmission technologies. This example shows use with a heat transmission technology:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/scenarios.yaml
   :language: yaml
   :dedent: 4
   :start-after: # heat_pipes-start
   :end-before: # heat_pipes-end

In the above example, heat pipes which distribute thermal energy in the network may be prone to dissipating heat in an unphysical way. I.e. given that they have distribution losses associated with them, in any given timestep, a link could produce and consume energy in the same timestep, losing energy to the atmosphere in both instances, but having a net energy transmission of zero. This might allow e.g. a CHP facility to overproduce heat to produce more cheap electricity, and have some way of dumping that heat. Enabling the ``force_async_flow`` parameter ensures that this does not happen.

-------------------------------
User-defined custom constraints
-------------------------------

It is possible to supply custom constraints, using the :ref:`math YAML syntax <custom_math>`.