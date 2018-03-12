================
Building a model
================

In short, a Calliope model works like this: **supply technologies** can take a **resource** from outside of the modeled system and turn it into a specific energy **carrier** in the system. The model specifies one or more **locations** along with the technologies allowed at those locations. **Transmission technologies** can move energy of the same carrier from one location to another, while **conversion technologies** can convert one carrier into another at the same location. **Demand technologies** remove energy from the system, while **storage technologies** can store energy at a specific location. Putting all of these possibilities together allows a modeller to specify as simple or as complex a model as necessary to answer a given research question.

In more technical terms, Calliope allows a modeller to define technologies with arbitrary characteristics by "inheriting" basic traits from a number of included base technologies -- ``supply``, ``supply_plus``, ``demand``, ``conversion``, ``conversion_plus``, and ``transmission``. The base technologies are described in more detail in the :ref:`reference section <technology_types>`.

-------------------------
Files that define a model
-------------------------

Calliope models are defined through YAML files, which are both human-readable and computer-readable (see :ref:`yaml_format` for details on this format), and CSV files (a simple tabular format) for time series data.

It makes sense to collect all files belonging to a model inside a single model directory. The layout of that directory typically looks roughly like this (``+`` denotes directories, ``-`` files):

.. code-block:: text

    + example_model
        + model_config
            - locations.yaml
            - techs.yaml
        + timeseries_data
            - solar_resource.csv
            - electricity_demand.csv
        - model.yaml
        - overrides.yaml

While it is possible to define the entire model in a single YAML file, it makes sense for purposes of readability to break up the definition into multiple files. A complete listing of the files in the included example models is available in :doc:`example_models`, showing what such a structure looks like in practice.

Inside the ``timeseries_data`` directory, timeseries are stored as CSV files (the location of this directory can be specified in the model configuration, e.g. in ``model.yaml``). For more details specifying time series data, refer to :ref:`configuration_timeseries` in the reference section.

In the above example, the files ``model.yaml``, ``locations.yaml`` and ``techs.yaml`` together are the model definition.

.. Note::

   The easiest way to create a new model is to use the ``calliope new`` command, which makes a copy of one of the built-in examples models::

   $ calliope new my_new_model

   This creates a new directory, ``my_new_model``, in the current working directory.

   By default, ``calliope new`` uses the national-scale example model as a template. To use a different template, you can specify the example model to use, e.g.: ``--template=urban_scale``.

-------------------------------
Model configuration (``model``)
-------------------------------

The model configuration specifies all aspects of the model to run. It is structured into several top-level headings (keys in the YAML file): ``model``, ``techs``, ``locations``, ``links``, and ``run``. We will discuss each of these in turn, starting with ``model``:

.. code-block:: yaml

    model:
        name: 'My energy model'
        mode: plan
        timeseries_data_path: 'timeseries_data'
        reserve_margin:
            power: 0
        subset_time: ['2005-01-01', '2005-01-05']

Besides the model's name (``name``) and the path for CSV time series data (``timeseries_data_path``), the most important part of the ``model`` section is ``mode``. A model can run either in planning mode (``plan``) or operational mode (``operate``).

In planning mode, constraints are given as upper and lower boundaries and the model decides on an optimal system configuration. In operational mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm (see :ref:`config_reference_model_wide` for the settings that control the receding horizon)

To specify a runnable operational model, capacities for all technologies at all locations must have be defined. This can be done by specifying ``energy_cap_equals``. In the absence of ``energy_cap_equals``, constraints given as ``energy_cap_max`` are assumed to be fixed in operational mode.

To speed up model runs, the above example specifies a time subset to run the model over only five days of time series data (``subset_time: ['2005-01-01', '2005-01-05']``)-- this is entirely optional. Usually, a full model will contain at least one year of data, but subsetting time can be useful to speed up a model for testing purposes.

For more details on the available options, see the national-scale example's :ref:`full file listing <examplemodels_nationalscale_settings>`, and :doc:`ref_config_listing`.

------------------------
Technologies (``techs``)
------------------------

The ``techs`` section in the model configuration specifies all of the model's technologies. In our current example, this is in a separate file, ``model_config/techs.yaml``, which is imported into the main ``model.yaml`` file alongside the file for locations described further below:

.. code-block:: yaml

    import:
        - 'model_config/techs.yaml'
        - 'model_config/locations.yaml'

The following example shows the definition of a ``ccgt`` technology, i.e. a combined cycle gas turbine that delivers electricity:

.. code-block:: yaml

    ccgt:
        essentials:
            name: 'Combined cycle gas turbine'
            color: '#FDC97D'
            parent: supply
            carrier_out: power
        constraints:
            resource: inf
            energy_eff: 0.5
            energy_cap_max: 40000  # kW
            energy_cap_max_systemwide: 100000  # kW
            energy_ramping: 0.8
            lifetime: 25
        costs:
            monetary:
                interest_rate: 0.10
                energy_cap: 750  # USD per kW
                om_con: 0.02  # USD per kWh

Each technology must specify some ``essentials``, most importantly a name, the abstract base technology it is inheriting from (``parent``), and its energy carrier (``carrier_out`` in the case of a ``supply`` technology). Specifying a ``color`` is optional but useful for using the built-in visualisation tools (see :doc:`analysing`).

The ``constraints`` section gives all constraints for the technology, such as allowed capacities, conversion efficiencies, the life time (used in levelised cost calculations), and the resource it consumes (in the above example, the resource is set to infinite via ``inf``). A full list of all possible constraints is given in :ref:`config_reference_constraints`.

The ``costs`` section gives costs for the technology. Calliope uses the concept of "cost classes" to allow accounting for more than just monetary costs. The above example specifies only the ``monetary`` cost class, but any number of other classes could be used, for example ``co2`` to account for emissions. See :ref:`config_reference_costs` for the full range of cost configuration options available.

See the :doc:`tutorials <tutorials>` and the :doc:`built-in examples <ref_example_models>` for some other examples of technology definitions, including the use of time series data.

Allowing for unmet demand
-------------------------

For a model to find a feasible solution even when in one or several timesteps, insufficient supply is available to meet demand, it is a good idea to always include a backstop technology that can provide extra supply to meet such unmet demand. In Calliope, such technologies can be added by defining a simple technology that inherits from the ``unmet_demand`` base technology:

.. code-block:: yaml

    unmet_demand_power:
        essentials:
            name: 'Unmet power demand'
            parent: unmet_demand
            carrier: power

This ``unmet_demand_power`` technology will automatically have a very high cost so that it is not used except when absolutely necessary.

----------------------------------------------
Locations and links (``locations``, ``links``)
----------------------------------------------

A model can specify any number of locations. These locations are linked together by transmission technologies. By consuming an energy carrier in one location and outputting it in another, linked location, transmission technologies allow resources to be drawn from the system at a different location from where they are brought into it.

The ``locations`` section specifies each location:

.. code-block:: yaml

    locations:
        region1:
            coordinates: {lat: 40, lon: -2}
            techs:
                unmet_demand_power:
                demand_power:
                ccgt:
                    constraints:
                        energy_cap_max: 30000

Locations can optionally specify ``coordinates`` (used in visualisation or to compute distance between them) and must specify ``techs`` allowed at that location. As seen in the example above, each allowed tech must be listed, and can optionally specify additional location-specific constraints. If given, location-specific constraints supersede any model-wide constraints a technology defines in the ``techs`` section for that location.

The ``links`` section specifies possible transmission links between locations in the form ``location1,location2``:

.. code-block:: yaml

    links:
        region1,region2:
            techs:
                ac_transmission:
                    constraints:
                        energy_cap_max: 10000

In the above example, an high-voltage AC transmission line is specified to connect ``region1`` with ``region2``. For this to work, a ``transmission`` technology called ``ac_transmission`` must have previously been defined in the model's ``techs`` section. There, it can be given model-wide constraints such as costs. As in the case of locations, the ``links`` section can specify per-link constraints that supersede any model-wide constraints.

The modeller can also specify a distance for each link, and use per-distance constraints and costs for transmission technologies. For more information on all available constraints and costs, see :ref:`config_reference_constraints` and :ref:`config_reference_costs`.

---------------------------
Run configuration (``run``)
---------------------------

The only required setting in the run configuration is the solver to use:

.. code-block:: yaml

    run:
        solver: glpk

Possible options for solver include ``glpk``, `gurobi`, ``cplex``, and ``cbc``. The interface to these solvers is done through the Pyomo library. Any solver compatible with Pyomo should work with Calliope.

For solvers with which Pyomo provides more than one way to interface, the additional ``solver_io`` option can be used. In the case of Gurobi, for example, it is usually fastest to use the direct Python interface:

.. code-block:: yaml

    run:
        solver: gurobi
        solver_io: python

Further optional settings, including debug settings, can be specified in the run configuration. More detail on these are given in :ref:`debugging_runs_config`.

Recommended settings to use with different solvers are detailed in :ref:`solver_options`.

---------
Overrides
---------

To make it easier to run a given model multiple times with slightly changed settings or constraints, for example, varying the cost of a key technology, it is possible to define and apply "override groups" in a separate file (in the above example, ``overrides.yaml``):

.. code-block:: yaml

    run1:
        model.subset_time: ['2005-01-01', '2005-01-31']
    run2:
        model.subset_time: ['2005-02-01', '2005-02-31']

Each group is given by a name (above, ``run1`` and ``run2`` and any number of model settings -- anything in the model configuration can be overridden by an override group). In the above example, the two runs specify different time subsets, so would run an otherwise identical model over two different periods of the time series data.

One or several override groups can be applied when running a model, as described in :doc:`running`. They can also be used to generate scripts that run many Calliope models sequentially or in parallel on a high-performance cluster, as detailed in :ref:`generating_scripts`.
