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

The first is ``ccgt`` (combined-cycle gas turbine), which serves as an example of a simple technology with an infinite source. Its only constraints are the cost of built capacity (``flow_cap``) and a constraint on its maximum built capacity.

.. figure:: images/supply.*
   :alt: Supply node

   The layout of a supply node, in this case ``ccgt``, which has an infinite source, a carrier conversion efficiency (:math:`flow_{eff}`), and a constraint on its maximum built :math:`flow_{cap}` (which puts an upper limit on :math:`flow_{out}`).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # ccgt-start
   :end-before: # ccgt-end

There are a few things to note. First, ``ccgt`` defines essential information: a name, a color (given as an HTML color code, for later visualisation), its parent, ``supply``, and its carrier_out, ``power``. It has set itself up as a power supply technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

.. Note:: There are technically no restrictions on the units used in model definitions. Usually, the units will be kW and kWh, alongside a currency like USD for costs. It is the responsibility of the modeler to ensure that units are correct and consistent. Some of the analysis functionality in the :mod:`~calliope.postprocess` module assumes that kW and kWh are used when drawing figure and axis labels, but apart from that, there is nothing preventing the use of other units.

The second technology is ``csp`` (concentrating solar power), and serves as an example of a complex supply_plus technology making use of:

* a finite source based on time series data
* built-in storage
* plant-internal losses (``parasitic_eff``)

.. figure:: images/supply_plus.*
   :alt: More complex node, with source storage

   The layout of a more complex node, in this case ``csp``, which makes use of most node-level functionality available.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # csp-start
   :end-before: # csp-end

Again, ``csp`` has the definitions for name, color, parent, and carrier_out. Its constraints are more numerous: it defines a maximum storage capacity (``storage_cap_max``), an hourly storage loss rate (``storage_loss``), then specifies that its source should be read from a file (more on that below). It also defines a carrier conversion efficiency of 0.4 and a parasitic efficiency of 0.9 (i.e., an internal loss of 0.1). Finally, the source collector area and the installed carrier conversion capacity are constrained to a maximum.

The costs are more numerous as well, and include monetary costs for all relevant components along the conversion from source to carrier (power): storage capacity, source collector area, source conversion capacity, carrier conversion capacity, and variable operational and maintenance costs. Finally, it also overrides the default value for the monetary interest rate.

Storage technologies
====================

The second location allows a limited amount of battery storage to be deployed to better balance the system. This technology is defined as follows:

.. figure:: images/storage.*
   :alt: Transmission node

   A storage node with an :math:`flow_{eff}` and :math:`storage_{loss}`.

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # battery-start
   :end-before: # battery-end

The contraints give a maximum installed generation capacity for battery storage together with a maximum ratio of flow capacity to storage capacity (``flow_cap_per_storage_cap_max``) of 4, which in turn limits the storage capacity. The ratio is the charge/discharge rate / storage capacity (a.k.a the battery `reservoir`). In the case of a storage technology, ``flow_in_eff`` applies on charging and ``flow_out_eff`` on discharging. In addition, storage technologies can lose stored carrier over time -- in this case, we set this loss to zero.

Other technologies
==================

Three more technologies are needed for a simple model. First, a definition of power demand:

.. figure:: images/demand.*
   :alt: Demand node

   A simple demand node.

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # demand-start
   :end-before: # demand-end

Power demand is a technology like any other. We will associate an actual demand time series with the demand technology later.

What remains to set up is a simple transmission technology. Transmission technologies (like conversion technologies) look different than other nodes, as they link the carrier at one location to the carrier at another (or, in the case of conversion, one carrier to another at the same location):

.. figure:: images/transmission.*
   :alt: Transmission node

   A simple transmission node with an :math:`flow_{eff}`.

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :dedent: 2
   :start-after: # transmission-start
   :end-before: # transmission-end

``ac_transmission`` has an efficiency of 0.85, so a loss during transmission of 0.15, as well as some cost definitions.

``free_transmission`` allows local power transmission from any of the csp facilities to the nearest location. As the name suggests, it applies no cost or efficiency losses to this transmission.

Locations
=========

In order to translate the model requirements shown in this section's introduction into a model definition, five locations are used: ``region1``, ``region2``, ``region1_1``, ``region1_2``, and ``region1_3``.

The technologies are set up in these locations as follows:

.. figure:: images/example_locations_national.*
   :alt: Locations and their technologies in the example model

   Locations and their technologies in the example model

Let's now look at the first location definition:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # region-1-start
   :end-before: # region-1-end

There are several things to note here:

* The location specifies a dictionary of technologies that it allows (``techs``), with each key of the dictionary referring to the name of technologies defined in our ``techs.yaml`` file. Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for both ``demand_power`` and ``ccgt``. For the latter, it simply sets a location-specific maximum capacity constraint. For ``demand_power``, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``sink`` option in the definition of the ``demand_power`` technology. Instead, this is done directly via a location-specific override. For this location, the file ``demand-1.csv`` is loaded and the column ``demand`` is taken (the text after the colon). If no column is specified, Calliope will assume that the column name matches the location name ``region1_1``. Note that in Calliope, a supply is positive and a demand is negative, so the stored CSV data will be negative.
* Coordinates are defined by latitude (``lat``) and longitude (``lon``), which will be used to calculate distance of transmission lines (unless we specify otherwise later on) and for location-based visualisation.

The remaining location definitions look like this:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # other-locs-start
   :end-before: # other-locs-end

``region2`` is very similar to ``region1``, except that it does not allow the ``ccgt`` technology. The three ``region1-`` locations are defined together, except for their location coordinates, i.e. they each get the exact same configuration. They allow only the ``csp`` technology, this allows us to model three possible sites for CSP plants.

For transmission technologies, the model also needs to know which locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../src/calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :dedent: 2
   :start-after: # links-start
   :end-before: # links-end

We are able to override constraints for transmission technologies at this point, such as the maximum capacity of the specific ``region1`` to ``region2`` link shown here.

Running the model
=================

We now take you through running the model in a :nbviewer_docs:`Jupyter notebook, which you can view here <_static/notebooks/national_scale.ipynb>`. After clicking on that link, you can also  download and run the notebook yourself (you will need to have Calliope installed).
