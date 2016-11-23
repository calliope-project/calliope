
========
Tutorial
========

Before going through this tutorial, it is recommended to have a brief look at the :doc:`components section <components>` to become familiar with the terminology and modeling approach used.

The tutorial is based on the built-in example model and explains the key steps necessary to set up and run a simple model. Refer to the other parts of the documentation for more detailed information on configuring and running more complex models.

The built-in example is simple on purpose, to show the key components of a Calliope model. It consists of two possible electricity supply technologies, an electricity demand at two locations, and a transmission technology linking the two. The diagram below gives an overview:

.. figure:: images/example_overview.*
   :alt: Overview of the built-in example model

   Overview of the built-in example model

------------------------
Supply-side technologies
------------------------

The example model defines two electricity supply technologies.

The first is ``ccgt`` (combined-cycle gas turbine), which serves as an example of a simple technology with an infinite resource. Its only constraints are the cost of built capacity (``e_cap``) and a constraint on its maximum built capacity.

.. figure:: images/node_simple.*
   :alt: Simple node

   The layout of a simple node, in this case ``ccgt``, which has an infinite resource, a resource conversion efficiency (``r_eff``), and a constraint on its maximum built ``e_cap`` (which puts an upper limit on ``es``).

The definition of this technology in the example model's configuration looks as follows:

.. literalinclude:: ../../calliope/example_model/model_config/techs.yaml
   :language: yaml
   :lines: 9-22

There are a few things to note. First, ``ccgt`` defines a name, a color (given as an HTML color code), and a stack_weight. These are used by the built-in analysis tools when analyzing model results. Second, it specifies its parent, ``supply``, and its carrier, ``power``, thus setting itself up as a power supply technology. This is followed by the definition of constraints and costs (the only cost class used is monetary, but this is where other "costs", such as emissions, could be defined).

.. Note:: There are technically no restrictions on the units used in model definitions. Usually, the units will be kW and kWh, alongside a currency like USD for costs. It is the responsibility of the modeler to ensure that units are correct and consistent. Some of the analysis functionality in the :mod:`~calliope.analysis` module assumes that kW and kWh are used when drawing figure and axis labels, but apart from that, there is nothing preventing the use of other units.

The second technology is ``csp`` (concentrating solar power), and serves as an example of a complex technology making use of:

* a finite resource based on time series data
* built-in storage
* plant-internal losses (``c_eff``)

.. figure:: images/node_intermediate.*
   :alt: More complex node but without a secondary resource

   The layout of a more complex node, in this case ``csp``, which makes use of most node-level functionality available, with the exception of a secondary resource.

This definition in the example model's configuration is more verbose:

.. literalinclude:: ../../calliope/example_model/model_config/techs.yaml
   :language: yaml
   :lines: 23-47

Again, ``csp`` has the definitions for name, color, stack_weight, parent, and carrier. Its constraints are more numerous: it defines a maximum storage time (``s_time.max``), an hourly storage loss rate (``s_loss``), then specifies that its resource should be read from a file (more on that below). It also defines an energy conversion efficiency of 0.4 and a carrier efficiency of 0.9 (i.e., an internal loss of 0.1). Finally, the resource collector area and the installed carrier conversion capacity are constrained to a maximum.

The costs are more numerous as well, and include monetary costs for all relevant components along the conversion from resource to carrier (power): storage capacity, resource collector area, resource conversion capacity, energy conversion capacity, and variable operational and maintenance costs. Finally, it also overrides the default value for the monetary interest rate.

------------------
Other technologies
------------------

Three more technologies are needed for a simple model. First, a definition of power demand and unmet power demand:

.. literalinclude:: ../../calliope/example_model/model_config/techs.yaml
   :language: yaml
   :lines: 51-58

Electricity demand is a technology like any other. We will associate an actual demand time series with the demand technology later. The parent of ``unmet_demand_power``, ``unmet_demand``, is a special kind of supply technology with an unlimited resource but very high cost. It allows a model to remain mathematically feasible even if insufficient supply is available to meet demand, and model results can easily be examined to verify whether there was any unmet demand. There is no requirement to include such a technology in a model, but it is useful to do so, since in its absence, an infeasible model would cause the solver to end with an error, returning no results for Calliope to analyze.

What remains to set up is a simple transmission technology:

.. literalinclude:: ../../calliope/example_model/model_config/techs.yaml
   :language: yaml
   :lines: 62-71

``hvac`` has an efficiency of 0.85, so a loss during transmission of 0.15, as well as some cost definitions.

Transmission technologies (like conversion technologies) look different than other nodes, as they link the carrier at one location to the carrier at another (or, in the case of conversion, one carrier to another at the same location). The following figure illustrates this for the example model's transmission technology:

.. figure:: images/node_transmission.*
   :alt: Transmission node

   A simple transmission node with an ``e_eff``.

---------
Locations
---------

In order to translate the model requirements shown in this section's introduction into a model definition, five locations are used: ``r1``, ``r2``, ``csp1``, ``csp2``, and ``csp3``.

The technologies are set up in these locations as follows:

.. figure:: images/example_locations.*
   :alt: Locations and their technologies in the example model

   Locations and their technologies in the example model

Let's now look at the first location definition:

.. literalinclude:: ../../calliope/example_model/model_config/locations.yaml
   :language: yaml
   :lines: 5-17

There are several things to note here:

* The location specifies a list of technologies that it allows (``techs``). Note that technologies listed here must have been defined elsewhere in the model configuration.
* It also overrides some options for both ``demand_power`` and ``ccgt``. For the latter, it simply sets a location-specific maximum capacity constraint. For ``demand_power``, the options set here are related to reading the demand time series from a CSV file. CSV is a simple text-based format that stores tables by comma-separated rows. Note that we did not define any ``r`` option in the definition of the ``demand_power`` technology. Instead, this is done directly via a location-specific override. For this location, the file ``demand-1.csv`` is loaded, and the demand is then scaled such that the demand peak is at the given value. Note that in Calliope, a supply is positive and a demand is negative, so the peak demand is actually a negative value. Finally, the ``x_map`` option allows us to read a CSV file with a single column named "demand" and tell Calliope to load data from that column for region ``r1``. This is necessary unless the column name(s) in the CSV file already correspond to the location names defined in the model configuration.

The remaining location definitions look like this:

.. literalinclude:: ../../calliope/example_model/model_config/locations.yaml
   :language: yaml
   :lines: 18-29

``r2`` is very similar to ``r1``, except that it does not allow the ``ccgt`` technology. The three ``csp`` locations are defined together, i.e. they each get the exact same configuration. They are ``within`` the location ``r1`` and allow only the ``csp`` technology, this allows us to model three possible sites for CSP plants within ``r1``.

Locations that do not specify a ``within`` are implicitly at the topmost level. Transmission between locations at the topmost level can only take place if transmission links are defined between them. On the other hand, locations which are specified as ``within`` another location can automatically and without any losses transmit energy to and from their parent location. In other words, a topmost location and all its contained locations together are implicitly assumed to be on a "copperplate" together. That means there are no transmission constraints and no transmission losses between these locations. Balancing of supply and demand takes place only at the topmost level.

For transmission technologies, the model also needs to know which top-level locations can be linked, and this is set up in the model configuration as follows:

.. literalinclude:: ../../calliope/example_model/model_config/locations.yaml
   :language: yaml
   :lines: 35-40

---------------------------
Files that define the model
---------------------------

The configuration definitions described above are in the YAML format, a simple human readable data serialization format, which is stored in text files with a .yaml (or .yml) extension. See :ref:`yaml_format` for details.

The layout of the model directory, which also includes the time series data in CSV format, is as follows (``+`` denotes directories, ``-`` files):

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

A complete listing of these configuration files is available in :doc:`example_model`.

Inside the ``data`` directory, time series are stored as CSV files (their location is configured inside ``model.yaml``). At a minimum, a model must always have a ``set_t.csv`` file which defines the model's timesteps. For more details on this and on time series data more generally, refer to :ref:`configuration_timeseries`.

The three files ``locations.yaml``, ``model.yaml``, and ``techs.yaml`` together are the model definition, and have been described above. There is one more YAML file, however: ``run.yaml``. This tells Calliope *how* to run the model given by the model definition, and will be described next. To run a model in Calliope, these two basic components -- a model definition and a run configuration -- are always required.

---------------------
The run configuration
---------------------

At its most basic, the run configuration simply specifies which model to run, which mode to run it in, and what solver to use. These three options are the required minimum. In the case of the example model, we also specify some output options. The output options only apply when the ``calliope run`` command-line tool is used to run the model (see below).

.. literalinclude:: ../../calliope/example_model/run.yaml
   :language: yaml
   :lines: 5-15

To speed up model runs, the built-in model's run configuration also specifies a time subset:

.. literalinclude:: ../../calliope/example_model/run.yaml
   :language: yaml
   :lines: 57

The included time series is hourly for a full year. The ``subset_t`` setting runs the model over only a subset of five days.

The full ``run.yaml`` file includes additional options, none of which are relevant for this tutorial. See the :ref:`full file listing <examplemodel_runsettings>` and the :doc:`section on the run configuration <run_configuration>` for more details on the available options.

Plan vs. operate
----------------

A Calliope model can either be run in planning mode (``mode: plan``) or operational mode (``mode: operate``). In planning mode, an optimization problem is solved to design an energy system that satisfies the given constraints.

In operational mode, all ``max`` constraints (such as ``e_cap.max``) are treated as fixed rather than as upper bounds. The resulting, fully defined energy system is then operated with a receding horizon control approach. The results are returned in exactly the same format as for planning mode results.

To specify a useful operational model, all locations will usually define overrides for options such as ``e_cap.max``, for all their allowed technologies.

For this tutorial, we are only using the planning mode.

-------------------------------------
Running a model and analyzing results
-------------------------------------

.. _tutorial_run_interactively:

Running interactively
---------------------

The most straightforward way to run a Calliope model is to do so in an interactive Python session.

An example which also demonstrates some of the analysis possibilities after running a model is given in the following Jupyter notebook. Note that you can download and run this notebook on your own machine (if both Calliope and the Jupyter Notebook are installed):

:nbviewer_docs:`Calliope interactive example notebook <_static/notebooks/tutorial.ipynb>`

Running with the command-line tool
----------------------------------

Another way to run a Calliope model is to use the command-line tool ``calliope run``. First, we create a new copy of the built-in example model, by using ``calliope new``::

   $ calliope new testmodel

This creates a new directory, ``testmodel``, in the current working directory. We can now run this model::

   $ calliope run testmodel/run.yaml

Because of the output options set in ``run.yaml``, model results will be stored as a set of CSV files in the directory ``Output``. Saving CSV files is an easy way to get results in a format suitable for further processing with other tools. In order to make use of Calliope's analysis functionality, results should be saved as a single NetCDF file instead, which comes with improved performance and handling. See :doc:`analysis` for more details, including the built-in functionality to read results from either CSV or NetCDF files, making them available for further analysis as described above (:ref:`tutorial_run_interactively`).
