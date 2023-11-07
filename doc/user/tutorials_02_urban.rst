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

The first two are ``supply_gas`` and ``supply_grid_power``, referring to the supply of ``gas`` (natural gas) and ``electricity``, respectively, from the national distribution system. These 'inifinitely' available national commodities can become carriers in the system, with the cost of their purchase being considered at supply, not conversion.

.. figure:: images/supply.*
   :alt: Simple node

   The layout of a simple supply technology, in this case ``supply_gas``, which has a source input and a carrier output. A carrier conversion efficiency (:math:`flow_{eff}`) can also be applied (although isn't considered for our supply technologies in this problem).

The definition of these technologies in the example model's configuration looks as follows:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # supply-start
   :end-before: # supply-end

The final supply technology is ``pv`` (solar photovoltaic power), which serves as an inflexible supply technology. It has a time-dependant source availablity, loaded from file, a maximum area over which it can capture its source (``area_use_max``) and a requirement that all available source must be used (``source_equals`` rather than ``source_max``). This emulates the reality of solar technologies: once installed, their production matches the availability of solar energy.

The efficiency of the DC to AC inverter (which occurs after conversion from source to carrier) is considered in ``parasitic_eff`` and the ``area_use_per_flow_cap`` gives a link between the installed area of solar panels to the installed capacity of those panels (i.e. kWp).

In most cases, domestic PV panels are able to export excess energy to the national grid. We allow this here by specifying an ``export_carrier``. Revenue for export will be considered on a per-location basis.

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # pv-start
   :end-before: # pv-end

Finally, the parent of the PV technology is not ``supply_plus``, but rather ``supply_power_plus``. We use this to show the possibility of an intermediate technology group, which provides the information on the energy carrier (``electricity``) and the ultimate abstract base technology (``supply_plus``):

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :start-after: # supply_power_plus-start
   :end-before: # supply_power_plus-end

Intermediate technology groups allow us to avoid repetition of technology information, be it in ``essentials``, ``constraints``, or ``costs``, by linking multiple technologies to the same intermediate group.

Conversion technologies
=======================

The example model defines two conversion technologies.

The first is ``boiler`` (natural gas boiler), which serves as an example of a simple conversion technology with one input carrier and one output carrier. Its only constraints are the cost of built capacity (``costs.monetary.flow_cap``), a constraint on its maximum built capacity (``constraints.flow_cap_max``), and a carrier conversion efficiency (``flow_out_eff``).

.. figure:: images/conversion.*
   :alt: Simple conversion node

   The layout of a simple node, in this case ``boiler``, which has one carrier input, one carrier output, a carrier conversion efficiency (:math:`flow_{eff}`), and a constraint on its maximum built :math:`flow_{cap}` (which puts an upper limit on :math:`flow_{out}`).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # boiler-start
   :end-before: # boiler-end

There are a few things to note. First, ``boiler`` defines a name, a color (given as an HTML color code), and a stack_weight. These are used by the built-in analysis tools when analyzing model results. Second, it specifies its parent, ``conversion``, its carrier_in ``gas``, and its carrier_out ``heat``, thus setting itself up as a gas to heat conversion technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

The second technology is ``chp`` (combined heat and power), and serves as an example of a possible conversion_plus technology making use of two output carriers.


.. figure:: images/conversion_plus.*
   :alt: More complex conversion technology, with multiple output carriers

   The layout of a more complex node, in this case ``chp``, which makes use of multiple output carriers.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # chp-start
   :end-before: # chp-end

.. seealso:: :ref:`conversion_plus`

Again, ``chp`` has the definitions for name, color, parent, and carrier_in/out. It now has an additional carrier (``carrier_out_2``) defined in its essential information, allowing a second carrier to be produced *at the same time* as the first carrier (``carrier_out``). The carrier ratio constraint tells us the ratio of carrier_out_2 to carrier_out that we can achieve, in this case 0.8 units of heat are produced every time a unit of electricity is produced. to produce these units of energy, gas is consumed at a rate of  ``flow_out(carrier_out) / flow_out_eff``, so gas consumption is only a function of power output.

As with the ``pv``, the ``chp`` an export eletricity. The revenue gained from this export is given in the file ``export_power.csv``, in which negative values are given per time step.

Demand technologies
===================

Electricity and heat demand are defined here:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # demand-start
   :end-before: # demand-end

Electricity and heat demand are technologies like any other. We will associate an actual demand time series with each demand technology later.

Transmission technologies
=========================

In this district, electricity and heat can be distributed between locations. Gas is made available in each location without consideration of transmission.

.. figure:: images/transmission.*
   :alt: Transmission node

   A simple transmission node with an :math:`flow_{eff}`.

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # transmission-start
   :end-before: # transmission-end

``power_lines`` has an efficiency of 0.95, so a loss during transmission of 0.05. ``heat_pipes`` has a loss rate per unit distance of 2.5%/unit distance (or ``flow_out_eff_per_distance`` of 97.5%). Over the distance between the two locations of 0.5km (0.5 units of distance), this translates to :math:`2.5^{0.5}` = 1.58% loss rate.


Locations
=========

In order to translate the model requirements shown in this section's introduction into a model definition, four locations are used: ``X1``, ``X2``, ``X3``, and ``N1``.

The technologies are set up in these locations as follows:

.. figure:: images/example_locations_urban.*
   :alt: Locations and their technologies in the example model

   Locations and their technologies in the urban-scale example model

Let's now look at the first location definition:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # X1-start
   :end-before: # X1-end

There are several things to note here:

* The location specifies a dictionary of technologies that it allows (``techs``), with each key of the dictionary referring to the name of technologies defined in our ``techs.yaml`` file. Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for both ``demand_electricity``, ``demand_heat``, and ``supply_grid_power``. For the latter, it simply sets a location-specific cost. For demands, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``sink`` option in the definition of these demands. Instead, this is done directly via a location-specific override. For this location, the files ``demand_heat.csv`` and ``demand_power.csv`` are loaded. As no column is specified (see :ref:`national scale example model <national_scale_example>`) Calliope will assume that the column name matches the location name ``X1``. Note that in Calliope, a supply is positive and a demand is negative, so the stored CSV data will be negative.
* Coordinates are defined by cartesian coordinates ``x`` and ``y``, which will be used to calculate distance of transmission lines (unless we specify otherwise later on) and for location-based visualisation. These coordinates are abstract, unlike latitude and longitude, and can be used when we don't know (or care) about the geographical location of our problem.
* An ``available_area`` is defined, which will limit the maximum area of all ``area_use`` technologies to the e.g. roof space available at our location. In this case, we just have ``pv``, but the case where solar thermal panels compete with photovoltaic panels for space, this would the sum of the two to the available area.

The remaining location definitions look like this:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # other-locs-start
   :end-before: # other-locs-end

``X2`` and ``X3`` are very similar to ``X1``, except that they do not connect to the national electricity grid, nor do they contain the ``chp`` technology. Specific ``pv`` cost structures are also given, emulating e.g. commercial vs. domestic feed-in tariffs.

``N1`` differs to the others by virtue of containing no technologies. It acts as a branching station for the heat network, allowing connections to one or both of ``X2`` and ``X3`` without double counting the pipeline from ``X1`` to ``N1``. Its definition look like this:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # N1-start
   :end-before: # N1-end

For transmission technologies, the model also needs to know which locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../src/calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # links-start
   :end-before: # links-end

The distance measure for the power line is larger than the straight line distance given by the coordinates of ``X1`` and ``X2``, so we can provide more information on non-direct routes for our distribution system. These distances will override any automatic straight-line distances calculated by coordinates.

Revenue by export
=================

Defined for both PV and CHP, there is the option to accrue revenue in the system by exporting electricity. This export is considered as a removal of the energy carrier ``electricity`` from the system, in exchange for negative cost (i.e. revenue). To allow this, ``export_carrier: electricity`` has been given under both technology definitions and an ``export`` value given under costs.

The revenue from PV export varies depending on location, emulating the different feed-in tariff structures in the UK for commercial and domestic properties. In domestic properties, the revenue is generated by simply having the installation (per kW installed capacity), as export is not metered. Export is metered in commercial properties, thus revenue is generated directly from export (per kWh exported). The revenue generated by CHP depends on the electricity grid wholesale price per kWh, being 80% of that. These revenue possibilities are reflected in the technologies' and locations' definitions.

Running the model
=================

We now take you through running the model in a :nbviewer_docs:`Jupyter notebook, which you can view here <_static/notebooks/urban_scale.ipynb>`. After clicking on that link, you can also  download and run the notebook yourself (you will need to have Calliope installed).
