================
Building a model
================

In short, a Calliope model works like this: **supply technologies** can take a **source** from outside of the modeled system and turn it into a specific **carrier** in the system. The model specifies one or more **locations** along with the technologies allowed at those locations. **Transmission technologies** can move the same carrier from one location to another, while **conversion technologies** can convert one carrier into another at the same location. **Demand technologies** remove carriers from the system through a **sink**, while **storage technologies** can store carriers at a specific location. Putting all of these possibilities together allows a modeller to specify as simple or as complex a model as necessary to answer a given research question.

In more technical terms, Calliope allows a modeller to define technologies with arbitrary characteristics by "inheriting" basic traits from a number of included base tech groups -- ``supply``, ``supply_plus``, ``demand``, ``conversion``, ``conversion_plus``, and ``transmission``. These groups are described in more detail in :ref:`abstract_base_tech_definitions`.

-----------
Terminology
-----------

The terminology defined here is used throughout the documentation and the model code and configuration files:

* **Technology**: a technology that produces, consumes, converts or transports carriers
* **Location**: a site which can contain multiple technologies and which may contain other locations for carrier balancing purposes
* **Source**: a source of commodity that can (or must) be used by a technology to introduce carriers into the system
* **Sink**: a commodity sink that can (or must) be used by a technology to remove carriers from the system
* **Carrier**: a carrier that groups technologies together into the same network, for example ``electricity`` or ``heat``.

As more generally in constrained optimisation, the following terms are also used:

* Parameter: a fixed coefficient that enters into model equations
* Variable: a variable coefficient (decision variable) that enters into model equations
* Set: an index in the algebraic formulation of the equations
* Constraint: an equality or inequality expression that constrains one or several variables

-------------------------
Files that define a model
-------------------------

Calliope models are defined through YAML files, which are both human-readable and computer-readable, and CSV files (a simple tabular format) for time series data.

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
        - scenarios.yaml

In the above example, the files ``model.yaml``, ``locations.yaml`` and ``techs.yaml`` together are the model definition. This definition could be in one file, but it is more readable when split into multiple. We use the above layout in the example models and in our research!

Inside the ``timeseries_data`` directory, timeseries are stored as CSV files. The location of this directory can be specified in the model configuration, e.g. in ``model.yaml``.

.. Note::

    The easiest way to create a new model is to use the ``calliope new`` command, which makes a copy of one of the built-in examples models::

    $ calliope new my_new_model

    This creates a new directory, ``my_new_model``, in the current working directory.

    By default, ``calliope new`` uses the national-scale example model as a template. To use a different template, you can specify the example model to use, e.g.: ``--template=urban_scale``.

.. seealso::

    :ref:`yaml_format`, :doc:`ref_example_models`, :ref:`configuration_timeseries`

.. _config:

--------------------------------
Model configuration (``config``)
--------------------------------

The model configuration specifies all aspects of the model to run.
It is structured into several top-level headings (keys in the YAML file): ``config``, ``parameters``, ``techs``, ``locations``, ``links``.
We will discuss each of these in turn, starting with ``config``:

.. code-block:: yaml

    config:
        init:
            name: 'My energy model'
            time_data_path: 'timeseries_data'
            time_subset: ['2005-01-01', '2005-01-05']
        build:
            mode: plan
        solve:
            solver: cbc

The ``init`` configuration items are accessed when you initialise your model (`calliope.Model(...)`).
The ``build`` configuration items are accessed when you build your optimisation problem (`calliope.Model.build(...)`).
The ``solve`` configuration items are accessed when you solve your optimisation problem (`calliope.Model.solve(...)`).

At each of these stages you can override what you have put in your YAML file (or if not in your YAML file, the default that Calliope uses), by providing additional keyword arguments on calling `calliope.Model` or its methods. E.g.,:

.. code-block:: python

    # Overriding `config.init` items in `calliope.Model`
    model = calliope.Model("path/to/model.yaml", time_subset=["2005-01", "2005-02"])
    # Overriding `config.build` items in `calliope.Model.build`
    model.build(ensure_feasibility=True)
    # Overriding `config.solve` items in `calliope.Model.solve`
    model.solve(save_logs="path/to/logs/dir")

None of the configuration options are _required_ as there is a default value for them all, but you will likely want to set `init.name`, `init.calliope_version`, `init.time_data_path`, `build.mode`, and `solve.solver`.

To test your model pipeline, `config.init.time_subset` is a good way to limit your model size by slicing the time dimension to a smaller range.

`config.build.mode`
^^^^^^^^^^^^^^^^^^^

In the ``build`` section we have ``mode``.
A model can run in ``plan``, ``operate``, or ``spores`` mode.
In `plan` mode, capacities are determined by the model, whereas in `operate` mode, capacities are fixed and the system is operated with a receding horizon control algorithm.
In `spores` mode, the model is first run in `plan` mode, then run `N` number of times to find alternative system configurations with similar monetary cost, but maximally different choice of technology capacity and location.

`config.solve.solver`
^^^^^^^^^^^^^^^^^^^^^
Possible options for solver include ``glpk``, ``gurobi``, ``cplex``, and ``cbc``.
The interface to these solvers is done through the Pyomo library. Any `solver compatible with Pyomo <https://pyomo.readthedocs.io/en/6.5.0/solving_pyomo_models.html#supported-solvers>`_ should work with Calliope.

For solvers with which Pyomo provides more than one way to interface, the additional ``solver_io`` option can be used.
In the case of Gurobi, for example, it is usually fastest to use the direct Python interface:

.. code-block:: yaml

    config:
        solve:
            solver: gurobi
            solver_io: python

.. note:: The opposite is currently true for CPLEX, which runs faster with the default ``solver_io``.

.. seealso::

    :ref:`config_reference_config`, :doc:`troubleshooting`, :ref:`solver_options`, :ref:`documentation on operate mode <operational_mode>`, :ref:`documentation on SPORES mode <spores_mode>`, :doc:`built-in examples <ref_example_models>`

-------------------------------------
Top-level parameters (``parameters``)
-------------------------------------

Some data is not indexed over technologies / nodes (as will be described in more detail below).
This data can be defined under the top-level key `parameters`.
This could be a single value:

.. code-block:: yaml

    parameters:
        my_param: 10

or (equivalent):

.. code-block:: yaml

    parameters:
        my_param:
            data: 10

which can then be accessed in the model inputs `model.inputs.my_param` and used in custom math as `my_param`.

Or it can be indexed over one or more model dimension(s):

.. code-block:: yaml

    parameters:
        my_indexed_param:
            data: 100
            index: [monetary]
            dims: costs
        my_multiindexed_param:
            data: [2, 10]
            index: [[monetary, electricity], [monetary, heat]]
            dims: [costs, carriers]

which can be accessed in the model inputs and custom math, e.g., `model.inputs.my_multiindexed_param.sel(costs="monetary")` and `my_multiindexed_param`.

You can also index over a new dimension:

.. code-block:: yaml

    parameters:
        my_indexed_param:
            data: { my_coord: 100 }
            dims: my_new_dim

Which will add the new dimension `my_new_dim` to your model: `model.inputs.my_new_dim` which you could choose to build a math component over:
`foreach: [my_new_dim]`.

.. warning::

    The `parameter` section should not be used for large datasets (e.g., indexing over the time dimension) as it will have a high memory overhead on loading the data.

------------------------
Technologies (``techs``)
------------------------

The ``techs`` section in the model configuration specifies all of the model's technologies. In our current example, this is in a separate file, ``model_config/techs.yaml``, which is imported into the main ``model.yaml`` file alongside the file for locations described further below:

.. code-block:: yaml

    import:
        - 'model_config/techs.yaml'
        - 'model_config/locations.yaml'

.. Note:: The ``import`` statement can specify a list of paths to additional files to import (the imported files, in turn, may include further files, so arbitrary degrees of nested configurations are possible). The ``import`` statement can either give an absolute path or a path relative to the importing file.

The following example shows the definition of a ``ccgt`` technology, i.e. a combined cycle gas turbine that delivers electricity:

.. code-block:: yaml

    ccgt:
        essentials:
            name: 'Combined cycle gas turbine'
            color: '#FDC97D'
            parent: supply
            carrier_out: power
        constraints:
            source: inf
            flow_out_eff: 0.5
            flow_cap_max: 40000  # kW
            flow_cap_max_systemwide: 100000  # kW
            flow_ramping: 0.8
            lifetime: 25
        costs:
            monetary:
                interest_rate: 0.10
                flow_cap: 750  # USD per kW
                flow_in: 0.02  # USD per kWh

Each technology must specify some ``essentials``, most importantly a name, the abstract base technology it is inheriting from (``parent``), and its carrier (``carrier_out`` in the case of a ``supply`` technology). Specifying a ``color`` is optional but useful for using the built-in visualisation tools (see :doc:`analysing`).

The ``constraints`` section gives all constraints for the technology, such as allowed capacities, conversion efficiencies, the life time (used in levelised cost calculations), and the resource it consumes (in the above example, the source is set to infinite via ``inf``).

The ``costs`` section gives costs for the technology. Calliope uses the concept of "cost classes" to allow accounting for more than just monetary costs. The above example specifies only the ``monetary`` cost class, but any number of other classes could be used, for example ``co2`` to account for emissions. Additional cost classes can be created simply by adding them to the definition of costs for a technology.

By default, only the ``monetary`` cost class is used in the objective function, i.e., the default objective is to minimize total costs.

Multiple cost classes can be considered in the objective by setting the `cost_class` key. It must be a dictionary of cost classes and their weights in the objective, e.g. :yaml:`objective_options: {'cost_class': {'monetary': 1, 'emissions': 0.1}}`. In this example, monetary costs are summed as usual and emissions are added to this, scaled by 0.1 (emulating a carbon price).

To use a different sense (minimize/maximize) you can set `sense`: :yaml:`objective_options: {'cost_class': ..., 'sense': 'minimize'}`.

To use a single alternative cost class, disabling the consideration of the default `monetary`, set the weight of the monetary cost class to zero to stop considering it and the weight of another cost class to a non-zero value, e.g. :yaml:`objective_options: {'cost_class': {'monetary': 0, 'emissions': 1}}`.

.. seealso::

    :ref:`config_reference_defaults`, :doc:`tutorials <tutorials>`, :doc:`built-in examples <ref_example_models>`

Allowing for unmet demand
^^^^^^^^^^^^^^^^^^^^^^^^^

For a model to find a feasible solution, supply must always be able to meet demand. To avoid the solver failing to find a solution, you can ensure feasibility:

.. code-block:: yaml

    config.build:
        ensure_feasibility: true

This will create an ``unmet_demand`` decision variable in the optimisation, which can pick up any mismatch between supply and demand, across all carriers. It has a very high cost associated with its use, so it will only appear when absolutely necessary.

.. note::
    When ensuring feasibility, you can also set a `big M value <https://en.wikipedia.org/wiki/Big_M_method>`_ (:yaml:`parameters.bigM`). This is the "cost" of unmet demand. It is possible to make model convergence very slow if bigM is set too high. default bigM is 1x10 :sup:`9`, but should be close to the maximum total system cost that you can imagine. This is perhaps closer to 1x10 :sup:`6` for urban scale models.

.. _configuration_timeseries:

----------------
Time series data
----------------

For parameters that vary in time, time series data can be added to a model in two ways:

* by reading in CSV files
* by passing ``pandas`` dataframes as arguments in ``calliope.Model`` called from a python session.

Reading in CSV files is possible from both the command-line tool as well running interactively with python (see :doc:`running` for details). However, passing dataframes as arguments in ``calliope.Model`` is possible only when running from a python session.

Reading in CSV files
^^^^^^^^^^^^^^^^^^^^
To read in CSV files, specify e.g., :yaml:`source: file=filename.csv` to pick the desired CSV file from within the configured timeseries data path (``model.time_data_path``).

By default, Calliope looks for a column in the CSV file with the same name as the location. It is also possible to specify a column to use when setting ``source`` per location, by giving the column name with a colon following the filename: :yaml:`source: file=filename.csv:column`

For example, a simple photovoltaic (PV) tech using a time series of hour-by-hour electricity generation data might look like this:

.. code-block:: yaml

    pv:
        essentials:
            name: 'Rooftop PV'
            color: '#B59C2B'
            parent: supply
            carrier_out: power
        constraints:
            source: file=pv_resource.csv
            flow_cap_max: 10000  # kW

By default, Calliope expects time series data in a model to be indexed by ISO 8601 compatible time stamps in the format ``YYYY-MM-DD hh:mm:ss``, e.g. ``2005-01-01 00:00:00``. This can be changed by setting :yaml:`model.time_format` based on ``strftime` directives <https://strftime.org/>`_, which defaults to ``"ISO8601"``.

For example, the first few lines of a CSV file, called ``pv_resource.csv`` giving a source potential for two locations might look like this, with the first column in the file always being read as the date-time index:

.. code-block:: text

    ,location1,location2
    2005-01-01 00:00:00,0,0
    2005-01-01 01:00:00,0,11
    2005-01-01 02:00:00,0,18
    2005-01-01 03:00:00,0,49
    2005-01-01 04:00:00,11,110
    2005-01-01 05:00:00,45,300
    2005-01-01 06:00:00,90,458

Reading in timeseries from ``pandas`` dataframes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When running models from python scripts or shells, it is also possible to pass timeseries directly as ``pandas`` dataframes. This is done by specifying :yaml:`source: df=tskey` where ``tskey`` is the key in a dictionary containing the relevant dataframes. For example, if the same timeseries as above is to be passed, a dataframe called ``pv_resource`` may be in the python namespace:

.. code-block:: python

    pv_resource

    t                     location1  location2
    2005-01-01 00:00:00           0          0
    2005-01-01 01:00:00           0         11
    2005-01-01 02:00:00           0         18
    2005-01-01 03:00:00           0         49
    2005-01-01 04:00:00          11        110
    2005-01-01 05:00:00          45        300
    2005-01-01 06:00:00          90        458

To pass this timeseries into the model, create a dictionary, called ``timeseries_dataframes`` here, containing all relevant timeseries identified by their ``tskey``. In this case, this has only one key, called ``pv_resource``:

.. code-block:: python

    timeseries_dataframes = {'pv_resource': pv_resource}

The keys in this dictionary must match the ``tskey`` specified in the YAML files. In this example, specifying :yaml:`source: df=pv_resource` will identify the ``pv_resource`` key in ``timeseries_dataframes``. All relevant timeseries must be put in this dictionary. For example, if a model contains three timeseries referred to in the configuration YAML files, called ``demand_1``, ``demand_2`` and ``pv_resource``, the ``timeseries_dataframes`` dictionary may look like

.. code-block:: python

    timeseries_dataframes = {'demand_1': demand_1,
                             'demand_2': demand_2,
                             'pv_resource': pv_resource}

where ``demand_1``, ``demand_2`` and ``pv_resource`` are dataframes of the relevant timeseries. The ``timeseries_dataframes`` can then be called in ``calliope.Model``:

.. code-block:: python

    model = calliope.Model('model.yaml', timeseries_dataframes=timeseries_dataframes)

Just like when using CSV files (see above), Calliope looks for a column in the dataframe with the same name as the location. It is also possible to specify a column to use when setting ``source`` per location, by giving the column name with a colon following the filename: :yaml:`source: df=tskey:column`.

The time series index must be ISO 8601 compatible time stamps and can be a standard ``pandas`` DateTimeIndex (see discussion above).


.. Note::

   * If a parameter is not explicit in time and space, it can be specified as a single value in the model definition (or, using location-specific definitions, be made spatially explicit). This applies both to parameters that never vary through time (for example, cost of installed capacity) and for those that may be time-varying (for example, a technology's available resource). However, each model must contain at least one time series.
   * You _cannot_ have a space around the ``=`` symbol when pointing to a timeseries file or dataframe key, i.e. :yaml:`source: file = filename.csv` is not valid.
   * If running from a command line interface (see :doc:`running`), timeseries must be read from CSV and cannot be passed from dataframes via ``df=...``.
   * It's possible to mix reading in from CSVs and dataframes, by setting some config values as ``file=...`` and some as ``df=...``.
   * The default value of ``timeseries_dataframes`` is ``None``, so if you want to read all timeseries in from CSVs, you can omit this argument. When running from command line, this is done automatically.

----------------------------------------------
Locations and links (``locations``, ``links``)
----------------------------------------------

A model can specify any number of locations. These locations are linked together by transmission technologies. By consuming a carrier in one location and outputting it in another, linked location, transmission technologies allow resources to be drawn from the system at a different location from where they are brought into it.

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
                        flow_cap_max: 30000

Locations can optionally specify ``coordinates`` (used in visualisation or to compute distance between them) and must specify ``techs`` allowed at that location. As seen in the example above, each allowed tech must be listed, and can optionally specify additional location-specific parameters (constraints or costs). If given, location-specific parameters supersede any group constraints a technology defines in the ``techs`` section for that location.

The ``links`` section specifies possible transmission links between locations in the form ``location1,location2``:

.. code-block:: yaml

    links:
        region1,region2:
            techs:
                ac_transmission:
                    constraints:
                        flow_cap_max: 10000
                    costs.monetary:
                        flow_cap: 100

In the above example, an high-voltage AC transmission line is specified to connect ``region1`` with ``region2``. For this to work, a ``transmission`` technology called ``ac_transmission`` must have previously been defined in the model's ``techs`` section. There, it can be given group constraints or costs. As in the case of locations, the ``links`` section can specify per-link parameters (constraints or costs) that supersede any model-wide parameters.

The modeller can also specify a distance for each link, and use per-distance constraints and costs for transmission technologies.

.. seealso::

    :ref:`config_reference_defaults`.

.. _building_overrides:

-----------------------
Scenarios and overrides
-----------------------

To make it easier to run a given model multiple times with slightly changed settings or constraints, for example, varying the cost of a key technology, it is possible to define and apply scenarios and overrides. "Overrides" are blocks of YAML that specify configurations that expand or override parts of the base model. "Scenarios" are combinations of any number of such overrides. Both are specified at the top level of the model configuration, as in this example ``model.yaml`` file:

.. code-block:: yaml

    scenarios:
        high_cost_2005: ["high_cost", "year2005"]
        high_cost_2006: ["high_cost", "year2006"]

    overrides:
        high_cost:
            techs.onshore_wind.costs.monetary.flow_cap: 2000
        year2005:
            model.time_subset: ['2005-01-01', '2005-12-31']
        year2006:
            model.time_subset: ['2006-01-01', '2006-12-31']

    config:
        ...

Each override is given by a name (e.g. ``high_cost``) and any number of model settings -- anything in the model configuration can be overridden by an override. In the above example, one override defines higher costs for an ``onshore_wind`` tech while the two other overrides specify different time subsets, so would run an otherwise identical model over two different periods of time series data.

One or several overrides can be applied when running a model, as described in :doc:`running`. Overrides can also be combined into scenarios to make applying them at run-time easier. Scenarios consist of a name and a list of override names which together form that scenario.

Scenarios and overrides can be used to generate scripts that run a single Calliope model many times, either sequentially, or in parallel on a high-performance cluster (see :ref:`generating_scripts`).

.. note::
    Overrides can also import other files. This can be useful if many overrides are defined which share large parts of model configuration, such as different levels of interconnection between model zones. See :ref:`imports_in_override_groups` for details.

.. seealso:: :ref:`generating_scripts`, :ref:`imports_in_override_groups`
