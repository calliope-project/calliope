=========
Tutorials
=========

Before going through these tutorials, it is recommended to have a brief look at the :doc:`components section <components>` to become familiar with the terminology and modeling approach used.

The tutorials are based on the built-in example models, they explain the key steps necessary to set up and run simple models. Refer to the other parts of the documentation for more detailed information on configuring and running more complex models.

The built-in examples are simple on purpose, to show the key components of a Calliope model.

The first part of the tutorial builds a model for part of a national grid, exhibiting the following Calliope functionality:

* Use of supply, supply_plus, demand, storage and transmission technologies
* Nested locations
* Multiple cost types

The second part of the tutorial builds a model for part of a district network, exhibiting the following Calliope functionality:

* Use of supply, demand, conversion, conversion_plus, and transmission technologies
* Use of multiple energy carriers
* Revenue generation, by carrier export

The third part of the tutorial extends the second part, exhibiting binary and integer decision variable functionality (extended an LP model to a MILP model)

.. _national_scale_example:

--------------------------
Tutorial 1: national scale
--------------------------

This example consists of two possible power supply technologies, a power demand at two locations, the possibility for battery storage at one of the locations, and a transmission technology linking the two. The diagram below gives an overview:

.. figure:: images/example_overview_national.*
   :alt: Overview of the built-in urban-scale example model

   Overview of the built-in national-scale example model

Supply-side technologies
========================

The example model defines two power supply technologies.

The first is ``ccgt`` (combined-cycle gas turbine), which serves as an example of a simple technology with an infinite resource. Its only constraints are the cost of built capacity (``energy_cap``) and a constraint on its maximum built capacity.

.. figure:: images/node_supply.*
   :alt: Supply node

   The layout of a supply node, in this case ``ccgt``, which has an infinite resource, a carrier conversion efficiency (:math:`energy_{eff}`), and a constraint on its maximum built :math:`energy_{cap}` (which puts an upper limit on :math:`energy_{prod}`).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 10-28

There are a few things to note. First, ``ccgt`` defines essential information: a name, a color (given as an HTML color code, for later visualisation), its parent, ``supply``, and its carrier_out, ``power``. It has set itself up as a power supply technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

.. Note:: There are technically no restrictions on the units used in model definitions. Usually, the units will be kW and kWh, alongside a currency like USD for costs. It is the responsibility of the modeler to ensure that units are correct and consistent. Some of the analysis functionality in the :mod:`~calliope.analysis` module assumes that kW and kWh are used when drawing figure and axis labels, but apart from that, there is nothing preventing the use of other units.

The second technology is ``csp`` (concentrating solar power), and serves as an example of a complex supply_plus technology making use of:

* a finite resource based on time series data
* built-in storage
* plant-internal losses (``parasitic_eff``)

.. figure:: images/node_supply_plus.*
   :alt: More complex node but without a secondary resource

   The layout of a more complex node, in this case ``csp``, which makes use of most node-level functionality available, with the exception of a secondary resource.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 29-52

Again, ``csp`` has the definitions for name, color, parent, and carrier_out. Its constraints are more numerous: it defines a maximum storage capacity (``storage_cap_max``), an hourly storage loss rate (``storage_loss``), then specifies that its resource should be read from a file (more on that below). It also defines a carrier conversion efficiency of 0.4 and a parasitic efficiency of 0.9 (i.e., an internal loss of 0.1). Finally, the resource collector area and the installed carrier conversion capacity are constrained to a maximum.

The costs are more numerous as well, and include monetary costs for all relevant components along the conversion from resource to carrier (power): storage capacity, resource collector area, resource conversion capacity, energy conversion capacity, and variable operational and maintenance costs. Finally, it also overrides the default value for the monetary interest rate.

Storage technologies
====================

The second location allows a limited amount of battery storage to be deployed to better balance the system. This technology is defined as follows:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 57-73

The contraints give a maximum installed generation capacity for battery storage together with a charge rate (``charge_rate``) of 4, which in turn limits the storage capacity. The charge rate is the charge/discharge rate / storage capacity (a.k.a the battery `resevoir`). In the case of a storage technology, ``energy_eff`` applies twice: on charging and discharging. In addition, storage technologies can lose stored energy over time -- in this case, we set this loss to zero.


Other technologies
==================

Three more technologies are needed for a simple model. First, a definition of power demand and unmet power demand:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 77-86

Power demand is a technology like any other. We will associate an actual demand time series with the demand technology later. The parent of ``unmet_demand_power``, ``unmet_demand``, is a special kind of supply technology with an unlimited resource but very high cost. It allows a model to remain mathematically feasible even if insufficient supply is available to meet demand, and model results can easily be examined to verify whether there was any unmet demand. There is no requirement to include such a technology in a model, but it is useful to do so, since in its absence, an infeasible model would cause the solver to end with an error, returning no results for Calliope to analyze.

What remains to set up is a simple transmission technologies:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 91-114

``ac_transmission`` has an efficiency of 0.85, so a loss during transmission of 0.15, as well as some cost definitions.

``free_transmission`` allows local power transmission from any of the ``csp`` facilities to the nearest location. As the name suggests, it applies no cost or efficiency losses to this transmission.

Transmission technologies (like conversion technologies) look different than other nodes, as they link the carrier at one location to the carrier at another (or, in the case of conversion, one carrier to another at the same location). The following figure illustrates this for the example model's transmission technology:

.. figure:: images/node_transmission.*
   :alt: Transmission node

   A simple transmission node with an :math:`energy_{eff}`.

Locations
=========

In order to translate the model requirements shown in this section's introduction into a model definition, five locations are used: ``region-1``, ``region-2``, ``region1-1``, ``region1-2``, and ``region1-3``.

The technologies are set up in these locations as follows:

.. figure:: images/example_locations_national.*
   :alt: Locations and their technologies in the example model

   Locations and their technologies in the example model

Let's now look at the first location definition:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :lines: 6-15

There are several things to note here:

* The location specifies a dictionary of technologies that it allows (``techs``), with each key of the dictionary referring to the name of technologies defined in our ``techs.yaml`` file. Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for both ``demand_power`` and ``ccgt``. For the latter, it simply sets a location-specific maximum capacity constraint. For ``demand_power``, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``resource`` option in the definition of the ``demand_power`` technology. Instead, this is done directly via a location-specific override. For this location, the file ``demand-1.csv`` is loaded and the column ``demand`` is taken (the text after the colon). If no column is specified, Calliope will assume that the column name matches the location name ``region1-1``. Note that in Calliope, a supply is positive and a demand is negative, so the stored CSV data will be negative.
* Coordinates are defined by latitude (``lat``) and longitude (``lon``), which will be used to calculate distance of transmission lines (unless we specify otherwise later on) and for location-based visualisation.

The remaining location definitions look like this:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :lines: 17-32

``region2`` is very similar to ``region1``, except that it does not allow the ``ccgt`` technology. The three ``region1-`` locations are defined together, except for their location coordinates, i.e. they each get the exact same configuration. They allow only the ``csp`` technology, this allows us to model three possible sites for CSP plants.

For transmission technologies, the model also needs to know which locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :lines: 38-48

We are able to override constraints for transmission technologies at this point, such as the maximum capacity of the specific ``region1`` to ``region2`` link shown here.

.. _urban_scale_example:

-----------------------
Tutorial 2: urban scale
-----------------------

This example consists of two possible sources of electricity, one possible source of heat, and one possible source of simultaneous heat and electricity. There are three locations, each describing a building, with transmission links between them. The diagram below gives an overview:

.. figure:: images/example_overview_urban.*
   :alt: Overview of the built-in urban-scale example model

   Overview of the built-in urban-scale example model

Supply technologies
===================

This example model defines three supply technologies.

The first two are ``supply_gas`` and ``supply_grid_power``, referring to the supply of ``gas`` (natural gas) and ``electricity``, respectively, from the national distribution system. These 'inifinitely' available national commodities can become energy carriers in the system, with the cost of their purchase being considered at supply, not conversion.

.. figure:: images/node_supply.*
   :alt: Simple node

   The layout of a simple supply technology, in this case ``supply_gas``, which has a resource input and a carrier output. A carrier conversion efficiency (:math:`energy_{eff}`) can also be applied (although isn't considered for our supply technologies in this problem).

The definition of these technologies in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 16-44

The final supply technology is ``pv`` (solar photovoltaic power), which serves as an inflexible supply technology. It has a time-dependant resource availablity, loaded from file, a maximum area over which it can capture its reosurce (``resource_area_max``) and a requirement that all available resource must be used (``force_resource: True``). This emulates the reality of solar technologies: once installed, their production matches the availability of solar energy.

The efficiency of the DC to AC inverter (which occurs after conversion from resource to energy carrier) is considered in ``parasitic_eff`` and the ``resource_area_per_energy_cap`` gives a link between the installed area of solar panels to the installed capacity of those panels (i.e. kWp).

In most cases, domestic PV panels are able to export excess energy to the national grid. We allow this here by specifying an ``export_carrier``. Revenue for export will be considered on a per-location basis.

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 48-66

Finally, the parent of the PV technology is not ``supply_plus``, but rather ``supply_power_plus``. We use this to show the possibility of an intermediate technology group, which provides the information on the energy carrier (``electricity``) and the ultimate abstract base technology (``supply_plus``):

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 5-9

Intermediate technology groups allow us to avoid repetition of technology information, be it in ``essentials``, ``constraints``, or ``costs``, by linking multiple technologies to the same intermediate group.

Conversion technologies
=======================

The example model defines two conversion technologies.

The first is ``boiler`` (natural gas boiler), which serves as an example of a simple conversion technology with one input carrier and one output carrier. Its only constraints are the cost of built capacity (``costs.monetary.energy_cap``), a constraint on its maximum built capacity (``constraints.energy_cap.max``), and an energy conversion efficiency (``energy_eff``).

.. figure:: images/node_conversion.*
   :alt: Simple conversion node

   The layout of a simple node, in this case ``boiler``, which has one carrier input, one carrier output, a carrier conversion efficiency (:math:`energy_{eff}`), and a constraint on its maximum built :math:`energy_{cap}` (which puts an upper limit on :math:`carrier_{prod}`).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 69-82

There are a few things to note. First, ``boiler`` defines a name, a color (given as an HTML color code), and a stack_weight. These are used by the built-in analysis tools when analyzing model results. Second, it specifies its parent, ``conversion``, its carrier_in ``gas``, and its carrier_out ``heat``, thus setting itself up as a gas to heat conversion technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

The second technology is ``chp`` (combined heat and power), and serves as an example of a possible conversion_plus technology making use of two output carriers.


.. figure:: images/node_conversion_plus.*
   :alt: More complex node but without a secondary resource

   The layout of a more complex node, in this case ``chp``, which makes use of multiple output carriers.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 86-106

Again, ``chp`` has the definitions for name, color, parent, and carrier_in/out. It now has an additional carrier (``carrier_out_2``) defined in its essential information, allowing a second carrier to be produced *at the same time* as the first carrier (``carrier_out``). The carrier ratio constraint tells us the ratio of carrier_out_2 to carrier_out that we can achieve, in this case 0.8 units of heat are produced every time a unit of electricity is produced. to produce these units of energy, gas is consumed at a rate of  ``carrier_prod(carrier_out) / energy_eff``, so gas consumption is only a function of power output.

As with the ``pv``, the ``chp`` an export eletricity. The revenue gained from this export is given in the file ``export_power.csv``, in which negative values are given per time step.

Demand technologies
===================

Electricity and heat demand, and their unmet_demand counterparts are defined here:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 110-132

Electricity and heat demand are technologies like any other. We will associate an actual demand time series with each demand technology later. The parent of ``unmet_demand_power`` and ``unmet_demand_heat``, ``unmet_demand``, is a special kind of supply technology with an unlimited resource but very high cost. It allows a model to remain mathematically feasible even if insufficient supply is available to meet demand, and model results can easily be examined to verify whether there was any unmet demand. There is no requirement to include such a technology in a model, but it is useful to do so, since in its absence, an infeasible model would cause the solver to end with an error, returning no results for Calliope to analyze.

Transmission technologies
=========================

In this district, electricity and heat can be distributed between locations. Gas is made available in each location without consideration of transmission.

.. figure:: images/node_transmission.*
   :alt: Transmission node

   A simple transmission node with an :math:`energy_{eff}`.

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 136-162

``power_lines`` has an efficiency of 0.95, so a loss during transmission of 0.05. ``heat_pipes`` has a loss rate per unit distance of 2.5%/unit distance (or ``energy_eff_per_distance`` of 97.5%). Over the distance between the two locations of 0.5km (0.5 units of distance), this translates to :math:`2.5^{0.5}` = 1.58% loss rate.


Locations
=========

In order to translate the model requirements shown in this section's introduction into a model definition, four locations are used: ``X1``, ``X2``, ``X3``, and ``N1``.

The technologies are set up in these locations as follows:

.. figure:: images/example_locations_urban.*
   :alt: Locations and their technologies in the example model

   Locations and their technologies in the urban-scale example model

Let's now look at the first location definition:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 2-16

There are several things to note here:

* The location specifies a dictionary of technologies that it allows (``techs``), with each key of the dictionary referring to the name of technologies defined in our ``techs.yaml`` file. Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for both ``demand_electricity``, ``demand_heat``, and ``supply_grid_power``. For the latter, it simply sets a location-specific cost. For demands, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``resource`` option in the definition of these demands. Instead, this is done directly via a location-specific override. For this location, the files ``demand_heat.csv`` and ``demand_power.csv`` are loaded. As no column is specified (see :ref:`national scale example model <national_scale_example>`) Calliope will assume that the column name matches the location name ``X1``. Note that in Calliope, a supply is positive and a demand is negative, so the stored CSV data will be negative.
* Coordinates are defined by cartesian coordinates ``x`` and ``y``, which will be used to calculate distance of transmission lines (unless we specify otherwise later on) and for location-based visualisation. These coordinates are abstract, unlike latitude and longitude, and can be used when we don't know (or care) about the geographical location of our problem.
* An ``available_area`` is defined, which will limit the maximum area of all ``resource_area`` technologies to the e.g. roof space available at our location. In this case, we just have ``pv``, but the case where solar thermal panels compete with photovoltaic panels for space, this would the sum of the two to the available area.

The remaining location definitions look like this:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 18-53

``X2`` and ``X3`` are very similar to ``X1``, except that they do not connect to the national electricity grid, nor do they contain the ``chp`` technology. Specific ``pv`` cost structures are also given, emulating e.g. commercial vs. domestic feed-in tariffs.

``N1`` differs to the others by virtue of containing no technologies. It acts as a branching station for the heat network, allowing connections to one or both of ``X2`` and ``X3`` without double counting the pipeline from ``X1`` to ``N1``. Its definition look like this:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 55-56

For transmission technologies, the model also needs to know which locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 58-69

The distance measure for the power line is larger than the straight line distance given by the coordinates of ``X1`` and ``X2``, so we can provide more information on non-direct routes for our distribution system. These distances will override any automatic straight-line distances calculated by coordinates.

Revenue by export
=================

Defined for both PV and CHP, there is the option to accrue revenue in the system by exporting electricity. This export is considered as a removal of the energy carrier ``electricity`` from the system, in exchange for negative cost (i.e. revenue). To allow this, ``carrier_export: electricity`` has been given under both technology definitions and an ``export`` value given under costs.

The revenue from PV export varies depending on location, emulating the different feed-in tariff structures in the UK for commercial and domestic properties. In domestic properties, the revenue is generated by simply having the installation (per kW installed capacity), as export is not metered. Export is metered in commercial properties, thus revenue is generated directly from export (per kWh exported). The revenue generated by CHP depends on the electricity grid wholesale price per kWh, being 80% of that. These revenue possibilities are reflected in the technologies' and locations' definitions.

.. _milp_example_model:

--------------------------------------------
Tutorial 3: Mixed Integer Linear Programming
--------------------------------------------
This example is based on the :ref:`urban scale example model <urban_scale_example>`, but with an override. An override file exists in which binary and integer decision variables are triggered, creating a MILP model, rather than the conventional Calliope LP model.

.. Warning::

   Integer and Binary variables are still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

Units
=====

The capacity of a technology is usually a continuous decision variable, which can be within the range of 0 and ``energy_cap_max`` (the maximum capacity of a technology). In this model, we introduce a unit limit on the CHP instead:

.. literalinclude:: ../../calliope/example_models/urban_scale/overrides.yaml
   :language: yaml
   :lines: 8-21

A unit maximum allows a discrete, integer number of CHP to be purchased, each having a capacity of ``energy_cap_per_unit``. Any of ``energy_cap_max``, ``energy_cap_min``, or ``energy_cap_equals`` are now ignored, in favour of ``units_max``, ``units_min``, or ``units_equals``. A useful feature unlocked by introducing this is the ability to set a minimum operating capacity which is *only* enforced when the technology is operating. In the LP model, ``energy_cap_min_use`` would force the technology to operate at least at that proportion of its maximum capacity at each time step. In this model, the newly introduced ``energy_cap_min_use`` of 0.2 will ensure that the output of the CHP is 20% of its maximum capacity in any time step in which it has a non-zero output.

Purchase cost
=============

The boiler does not have a unit limit, it still utilises the continuous variable for its capacity. However, we have introduced a ``purchase`` cost:

.. literalinclude:: ../../calliope/example_models/urban_scale/overrides.yaml
   :language: yaml
   :lines: 17-21

By introducing this, the boiler now has a binary decision variable associated with it, which is 1 if the boiler has a non-zero ``energy_cap`` (i.e. the optimisation results in investement in a boiler) and 0 if the capacity is 0. The purchase cost is applied to the binary result, providing a fixed cost on purchase of the technology, irrespective of the technology size. In physical terms, this may be associated with the cost of pipework, land purchase, etc. The purchase cost is also imposed on the CHP, which is applied to the number of integer CHP units in which the solver chooses to invest.

MILP functionality can be easily applied, but convergence is slower as a result of integer/binary variables. It is recommended to use a commercial solver (e.g. Gurobi, CPLEX) if you wish to utilise these variables outside this example model.
