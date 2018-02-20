
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

The first is ``ccgt`` (combined-cycle gas turbine), which serves as an example of a simple technology with an infinite resource. Its only constraints are the cost of built capacity (``e_cap``) and a constraint on its maximum built capacity.

.. figure:: images/node_supply.*
   :alt: Supply node

   The layout of a supply node, in this case ``ccgt``, which has an infinite resource, a carrier conversion efficiency (:math:`e_{eff}`), and a constraint on its maximum built :math:`e_{cap}` (which puts an upper limit on :math:`e_{prod}`).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 10-23

There are a few things to note. First, ``ccgt`` defines a name, a color (given as an HTML color code), and a stack_weight. These are used by the built-in analysis tools when analyzing model results. Second, it specifies its parent, ``supply``, and its carrier_out, ``power``, thus setting itself up as a power supply technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

.. Note:: There are technically no restrictions on the units used in model definitions. Usually, the units will be kW and kWh, alongside a currency like USD for costs. It is the responsibility of the modeler to ensure that units are correct and consistent. Some of the analysis functionality in the :mod:`~calliope.analysis` module assumes that kW and kWh are used when drawing figure and axis labels, but apart from that, there is nothing preventing the use of other units.

The second technology is ``csp`` (concentrating solar power), and serves as an example of a complex supply_plus technology making use of:

* a finite resource based on time series data
* built-in storage
* plant-internal losses (``p_eff``)

.. figure:: images/node_supply_plus.*
   :alt: More complex node but without a secondary resource

   The layout of a more complex node, in this case ``csp``, which makes use of most node-level functionality available, with the exception of a secondary resource.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 24-48

Again, ``csp`` has the definitions for name, color, stack_weight, parent, and carrier_out. Its constraints are more numerous: it defines a maximum storage time (``s_time.max``), an hourly storage loss rate (``s_loss``), then specifies that its resource should be read from a file (more on that below). It also defines a carrier conversion efficiency of 0.4 and a parasitic efficiency of 0.9 (i.e., an internal loss of 0.1). Finally, the resource collector area and the installed carrier conversion capacity are constrained to a maximum.

The costs are more numerous as well, and include monetary costs for all relevant components along the conversion from resource to carrier (power): storage capacity, resource collector area, resource conversion capacity, energy conversion capacity, and variable operational and maintenance costs. Finally, it also overrides the default value for the monetary interest rate.

Storage technologies
====================

The second location allows a limited amount of battery storage to be deployed to better balance the system. This technology is defined as follows:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 53-66

The contraints give a maximum installed generation capacity for battery storage together with a charge rate (C-rate) of 4, which in turn limits the storage capacity. In the case of a storage technology, ``e_eff`` applies twice: on charging and discharging. In addition, storage technologies can lose stored energy over time -- in this case, we set this loss to zero.


Other technologies
==================

Three more technologies are needed for a simple model. First, a definition of power demand and unmet power demand:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 70-77

Power demand is a technology like any other. We will associate an actual demand time series with the demand technology later. The parent of ``unmet_demand_power``, ``unmet_demand``, is a special kind of supply technology with an unlimited resource but very high cost. It allows a model to remain mathematically feasible even if insufficient supply is available to meet demand, and model results can easily be examined to verify whether there was any unmet demand. There is no requirement to include such a technology in a model, but it is useful to do so, since in its absence, an infeasible model would cause the solver to end with an error, returning no results for Calliope to analyze.

What remains to set up is a simple transmission technology:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/techs.yaml
   :language: yaml
   :lines: 82-91

``ac_transmission`` has an efficiency of 0.85, so a loss during transmission of 0.15, as well as some cost definitions.

Transmission technologies (like conversion technologies) look different than other nodes, as they link the carrier at one location to the carrier at another (or, in the case of conversion, one carrier to another at the same location). The following figure illustrates this for the example model's transmission technology:

.. figure:: images/node_transmission.*
   :alt: Transmission node

   A simple transmission node with an :math:`e_{eff}`.

Locations
=========

In order to translate the model requirements shown in this section's introduction into a model definition, five locations are used: ``r1``, ``r2``, ``csp1``, ``csp2``, and ``csp3``.

The technologies are set up in these locations as follows:

.. figure:: images/example_locations_national.*
   :alt: Locations and their technologies in the example model

   Locations and their technologies in the example model

Let's now look at the first location definition:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :lines: 5-16

There are several things to note here:

* The location specifies a list of technologies that it allows (``techs``). Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for both ``demand_power`` and ``ccgt``. For the latter, it simply sets a location-specific maximum capacity constraint. For ``demand_power``, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``r`` option in the definition of the ``demand_power`` technology. Instead, this is done directly via a location-specific override. For this location, the file ``demand-1.csv`` is loaded, and the demand is then scaled such that the demand peak is at the given value. Note that in Calliope, a supply is positive and a demand is negative, so the peak demand is actually a negative value. Finally, the ``x_map`` option allows us to read a CSV file with a single column named "demand" and tell Calliope to load data from that column for region ``r1``. This is necessary unless the column name(s) in the CSV file already correspond to the location names defined in the model configuration.

The remaining location definitions look like this:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :lines: 18-29

``r2`` is very similar to ``r1``, except that it does not allow the ``ccgt`` technology. The three ``csp`` locations are defined together, i.e. they each get the exact same configuration. They are ``within`` the location ``r1`` and allow only the ``csp`` technology, this allows us to model three possible sites for CSP plants within ``r1``.

Locations that do not specify a ``within`` are implicitly at the topmost level. Transmission between locations at the topmost level can only take place if transmission links are defined between them. On the other hand, locations which are specified as ``within`` another location can automatically and without any losses transmit energy to and from their parent location. In other words, a topmost location and all its contained locations together are implicitly assumed to be on a "copperplate" together. That means there are no transmission constraints and no transmission losses between these locations. Balancing of supply and demand takes place only at the topmost level.

For transmission technologies, the model also needs to know which top-level locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../calliope/example_models/national_scale/model_config/locations.yaml
   :language: yaml
   :lines: 35-39

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

The first two are ``national_gas`` and ``national_grid``, referring to the supply of ``gas`` (natural gas) and ``power`` (electricity), respectively, from the national distribution system. These 'inifinitely' available national commodities can become energy carriers in the system, with the cost of their purchase being considered at supply, not conversion.

.. figure:: images/node_supply.*
   :alt: Simple node

   The layout of a simple node, in this case ``boiler``, which has one carrier input, one carrier output, a carrier conversion efficiency (:math:`e_{eff}`), and a constraint on its maximum built :math:`e_{cap}` (which puts an upper limit on :math:`e_{prod}`).

The definition of these technologies in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 7-33

The final supply technology is ``pv`` (solar photovoltaic power), which serves as a inflexible supply technology. It is simple to define, other than having a time-dependant resource availablity, loaded from file. Additionally, it is constrained by available area, which is the rooftop area of the locations in this example.

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 35-53

Conversion technologies
=======================

The example model defines two conversion technologies.

The first is ``boiler`` (natural gas boiler), which serves as an example of a simple conversion technology with one input carrier and one output carrier. Its only constraints are the cost of built capacity (``e_cap``) and a constraint on its maximum built capacity.

.. figure:: images/node_conversion.*
   :alt: Simple node

   The layout of a simple node, in this case ``boiler``, which has one carrier input, one carrier output, a carrier conversion efficiency (:math:`e_{eff}`), and a constraint on its maximum built :math:`e_{cap}` (which puts an upper limit on :math:`e_{prod}`).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 54-65

There are a few things to note. First, ``boiler`` defines a name, a color (given as an HTML color code), and a stack_weight. These are used by the built-in analysis tools when analyzing model results. Second, it specifies its parent, ``conversion``, its carrier_in ``gas``, and its carrier_out ``heat``, thus setting itself up as a gasto heat conversion technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

The second technology is ``chp`` (combined heat and power), and serves as an example of a possible conversion_plus technology making use of two output carriers.


.. figure:: images/node_conversion_plus.*
   :alt: More complex node but without a secondary resource

   The layout of a more complex node, in this case ``chp``, which makes use of multiple output carriers.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 67-87

Again, ``chp`` has the definitions for name, color, stack_weight, parent, and carrier_in. Its constraints are no more numerous: it still only defines a carrier conversion efficiency and maximum carrier conversion capacity.


Demand technologies
===================

Electricity and heat demand, and their unmet_demand counterparts are defined here:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 89-111

Electricity and heat demand are a technologies like any other. We will associate an actual demand time series with each demand technology later. The parent of ``unmet_demand_power`` and ``unmet_demand_heat``, ``unmet_demand``, is a special kind of supply technology with an unlimited resource but very high cost. It allows a model to remain mathematically feasible even if insufficient supply is available to meet demand, and model results can easily be examined to verify whether there was any unmet demand. There is no requirement to include such a technology in a model, but it is useful to do so, since in its absence, an infeasible model would cause the solver to end with an error, returning no results for Calliope to analyze.

Transmission technologies
=========================

In this district, electricity and heat can be transmitted between two locations. Gas is made available in each location without consideration of transmission.

.. figure:: images/node_transmission.*
   :alt: Transmission node

   A simple transmission node with an :math:`e_{eff}`.

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/techs.yaml
   :language: yaml
   :lines: 113-138

``power_lines`` has an efficiency of 0.95, so a loss during transmission of 0.05. ``heat_pipes`` has a loss rate per unit distance of 2.5%/km. Over the distance between the two locations of 0.5km, this translates to 1.25% loss rate.


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
   :lines: 5-15

There are several things to note here:

* The location specifies a list of technologies that it allows (``techs``). Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for ``demand_power`` and ``ccgt``. For the latter, it simply sets a location-specific maximum capacity constraint. For ``demand_power``, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``r`` option in the definition of the ``demand_power`` technology. Instead, this is done directly via a location-specific override. For this location, the file ``demand-1.csv`` is loaded, and the demand is then scaled such that the demand peak is at the given value. Note that in Calliope, a supply is positive and a demand is negative, so the peak demand is actually a negative value. Finally, the ``x_map`` option allows us to read a CSV file with a single column named "demand" and tell Calliope to load data from that column for region ``r1``. This is necessary unless the column name(s) in the CSV file already correspond to the location names defined in the model configuration.

The remaining location definitions look like this:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 17-51

``X2`` and ``X3`` are very similar to ``X1``, except that they do not connect to the national grid, nor do they contain the ``chp`` technology.

``N1`` differs to the others by virtue of containing no technologies. It acts as a branching station for the heat network, allowing connections to one or both of ``X2`` and ``X3`` without double counting the pipeline from ``X1`` to ``N1``. Its definition look like this:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 50-51

For transmission technologies, the model also needs to know which top-level locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../calliope/example_models/urban_scale/model_config/locations.yaml
   :language: yaml
   :lines: 53-68

Revenue by export
=================

Defined for both PV and CHP, there is the option to accrue revenue in the system by exporting electricity. This export is considered as a removal of the energy carrier ``power`` from the system, in exchange for negative cost (i.e. revenue). To allow this, ``export: true`` has been given under both technology definitions and an ``export`` value given under costs.

The revenue from PV export varies depending on location, emulating the different feed-in tariff structures in the UK for commercial and domestic properties. In domestic properties, the revenue is generated by simply having the installation (per kW installed capacity), as export is not metered. Export is metered in commercial properties, thus revenue is generated directly from export (per kWh exported). The revenue generated by CHP depends on the electricity grid wholesale price per kWh, being 80% of that. These revenue possibilities are reflected in the technologies' and locations' definitions.

--------------------------------------------
Tutorial 3: Mixed Integer Linear Programming
--------------------------------------------
This example is based on the :ref:`urban scale example model <urban_scale_example>`, but with an updated run configuration. This run configuration provides technology overrides which trigger binary and integer decision variables, creating a MILP model, rather than the conventional Calliope LP model.

.. Warning::

   Integer and Binary variables are still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

Units
=====

The capacity of a technology is a continuous decision variable, which can be within the range of 0 and ``e_cap.max[y, x]`` (the maximum capacity of a technology ``y`` at location ``x``). In this run, we introduce a unit limit on the CHP instead:

.. literalinclude:: ../../calliope/example_models/urban_scale/run_milp.yaml
   :language: yaml
   :lines: 44-54

A unit maximum allows a discrete, integer number of CHP to be purchased, each having a capacity of ``e_cap_per_unit``. Any of ``e_cap.max``, ``e_cap.min``, or ``e_cap.equals`` are ignored, in favour of ``units.max``, ``units.min``, or ``units.equals``. A useful feature unlocked by introducing this is the ability to set a minimum operating capacity which is *only* enforced when the technology is operating. In the LP model, ``e_cap_min_use`` would force the technology to operate at least at that proportion of its maximum capacity at each time step. In this model, the newly introduced ``e_cap_min_use`` of 0.2 will ensure that the output of the CHP is 20% of its maximum capacity in any time step in which it has a finite output.

Purchase cost
=============

The boiler does not have a unit limit, it still utilises the continuous variable for its capacity. However, we have introduced a `purchase` cost:

.. literalinclude:: ../../calliope/example_models/urban_scale/run_milp.yaml
   :language: yaml
   :lines: 55-59

By introducing this, the boiler now has a binary decision variable associated with it, which is 1 if the boiler has a non-zero capacity (i.e. the optimisation results in investement in a boiler) and 0 if the capacity is 0. The purchase cost is applied to the binary result, providing a fixed cost on purchase of the technology, irrespective of the technology size. In physical terms, this may be associated with the cost of pipework, land purchase, etc. The purchase cost is also imposed on the CHP, which is applied to the number of integer CHP units which are invested in.

MILP functionality can be easily applied, but convergence is slower as a result of integer/binary variables. It is recommended to use a commercial solver (e.g. Gurobi, CPLEX) if you wish to utilise these variables outside this example model.

---------------------------
Files that define the model
---------------------------

For all Calliope models, including the examples discussed above, the model definitions in through YAML files, which are simple human-readable text files (YAML is a human readable data serialization format). They are stored with a ``.yaml`` (or ``.yml``) extension. See :ref:`yaml_format` for details.

Typically, we want to collect all files belonging to a model inside a model directory. In the national-scale example describe above, the layout of that directory, which also includes the time series data in CSV format, is as follows (``+`` denotes directories, ``-`` files):

.. code-block:: text

   + example_model
      + model_config
         + data
            - csp_r.csv
            - demand-1.csv
            - demand-2.csv
            - set_t.csv
         - locations.yaml
         - model.yaml
         - techs.yaml
      - run.yaml

The urban-scale example follows a similar layout. A complete listing of the files in all example models is available in :doc:`example_models`.

Inside the ``data`` directory, time series are stored as CSV files (their location is configured inside ``model.yaml``). At a minimum, a model must always have a ``set_t.csv`` file which defines the model's timesteps. For more details on this and on time series data more generally, refer to :ref:`configuration_timeseries`.

The three files ``locations.yaml``, ``model.yaml``, and ``techs.yaml`` together are the model definition, and have been described above. There is one more YAML file, however: ``run.yaml``. This tells Calliope *how* to run the model given by the model definition, and will be described next. To run a model in Calliope, these two basic components -- a model definition and a run configuration -- are always required.

---------------------
The run configuration
---------------------

At its most basic, the run configuration simply specifies which model to run, which mode to run it in, and what solver to use. These three options are the required minimum. In the case of the example models, we also specify some output options. The output options only apply when the ``calliope run`` command-line tool is used to run the model (see below). In the national-scale example:

.. literalinclude:: ../../calliope/example_models/national_scale/run.yaml
   :language: yaml
   :lines: 5-15

To speed up model runs, the national-scale example model's run configuration also specifies a time subset:

.. literalinclude:: ../../calliope/example_models/national_scale/run.yaml
   :language: yaml
   :lines: 57

The included time series is hourly for a full year. The ``subset_t`` setting runs the model over only a subset of five days.

The full ``run.yaml`` file includes additional options, none of which are relevant for this tutorial. See the :ref:`full file listing <examplemodels_nationalscale_runsettings>` for the national-scale example and the :doc:`section on the run configuration <run_configuration>` for more details on the available options.

Plan vs. operate
================

A Calliope model can either be run in planning mode (``mode: plan``) or operational mode (``mode: operate``). In planning mode, an optimization problem is solved to design an energy system that satisfies the given constraints.

In operational mode, all ``max`` constraints (such as ``e_cap.max``) are treated as fixed rather than as upper bounds. The resulting, fully defined energy system is then operated with a receding horizon control approach. The results are returned in exactly the same format as for planning mode results.

To specify a runnable operational model, capacities for all technologies at all locations would have to be defined. This can be done by specifying ``e_cap.equals``. In the absence of ``e_cap.equals``, ``e_cap.max`` is assumed to be fixed.

In this tutorial section, we are only demonstrating the planning mode.

-------------------------------------
Running a model and analyzing results
-------------------------------------

.. _tutorial_run_interactively:

Running interactively
=====================

The most straightforward way to run a Calliope model is to do so in an interactive Python session.

An example which also demonstrates some of the analysis possibilities after running a model is given in the following Jupyter notebook, based on the national-scale example model. Note that you can download and run this notebook on your own machine (if both Calliope and the Jupyter Notebook are installed):

:nbviewer_docs:`Calliope interactive national-scale example notebook <_static/notebooks/tutorial.ipynb>`

Running with the command-line tool
==================================

Another way to run a Calliope model is to use the command-line tool ``calliope run``. First, we create a new copy of the built-in national-scale example model, by using ``calliope new``::

   $ calliope new testmodel

.. Note:: By default, ``calliope new`` uses the national-scale example model as a template. To use a different template, you can specify the example model to use, e.g.: ``--template=UrbanScale``.

This creates a new directory, ``testmodel``, in the current working directory. We can now run this model::

   $ calliope run testmodel/run.yaml

Because of the output options set in ``run.yaml``, model results will be stored as a set of CSV files in the directory ``Output``. Saving CSV files is an easy way to get results in a format suitable for further processing with other tools. In order to make use of Calliope's analysis functionality, results should be saved as a single NetCDF file instead, which comes with improved performance and handling.

See :doc:`running` for more on how to run a model and then retrieve results from it. See :doc:`analysis` for more details on analyzing results, including the built-in functionality to read results from either CSV or NetCDF files, making them available for further analysis as described above (:ref:`tutorial_run_interactively`).
