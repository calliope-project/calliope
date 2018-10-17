----------------------
Advanced functionality
----------------------

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

.. _configuration_timeseries:

Time series data
----------------

.. Note::

   If a parameter is not explicit in time and space, it can be specified as a single value in the model definition (or, using location-specific definitions, be made spatially explicit). This applies both to parameters that never vary through time (for example, cost of installed capacity) and for those that may be time-varying (for example, a technology's available resource).


For parameters that vary in time, time series data can be read from CSV files, by specifying ``resource: file=filename.csv`` to pick the desired CSV file from within the configured timeseries data path (``model.timeseries_data_path``).

By default, Calliope looks for a column in the CSV file with the same name as the location. It is also possible to specify a column too use when setting ``resource`` per location, by giving the column name with a colon following the filename: ``resource: file=filename.csv:column``

All time series data in a model must be indexed by ISO 8601 compatible time stamps (usually in the format ``YYYY-MM-DD hh:mm:ss``, e.g. ``2005-01-01 00:00:00``), i.e., the first column in the CSV file must be time stamps.

For example, the first few lines of a CSV file giving a resource potential for two locations might look like this:

.. code-block:: text

    ,location1,location2
    2005-01-01 00:00:00,0,0
    2005-01-01 01:00:00,0,11
    2005-01-01 02:00:00,0,18
    2005-01-01 03:00:00,0,49
    2005-01-01 04:00:00,11,110
    2005-01-01 05:00:00,45,300
    2005-01-01 06:00:00,90,458

.. _time_clustering:

Time resolution adjustment
--------------------------

Models have a default timestep length (defined implicitly by the timesteps of the model's time series data). This default resolution can be adjusted over parts of the dataset by specifying time resolution adjustment in the model configuration, for example:

.. code-block:: yaml

    model:
        time:
            function: resample
            function_options: {'resolution': '6H'}

In the above example, this would resample all time series data to 6-hourly timesteps.

Calliope's time resolution adjustment functionality allows running a function that can perform arbitrary adjustments to the time series data in the model.

The available options include:

1. Uniform time resolution reduction through the ``resample`` function, which takes a `pandas-compatible rule describing the target resolution <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html>`_ (see above example).

2. Deriving representative days from the input time series, by applying the clustering method implemented in :mod:`calliope.time.clustering`, for example:

.. code-block:: yaml

    model:
        time:
            function: apply_clustering
            function_options:
                clustering_func: kmeans
                how: mean
                k: 20

When using representative days, a number of additional constraints are added, based on the study undertaken by `Kotzur et al <https://doi.org/10.1016/j.apenergy.2018.01.023>`_. These constraints require a new decision variable ``storage_inter_cluster``, which tracks storage between all the dates of the original timeseries. This particular functionality can be disabled by including ``storage_inter_cluster: false`` in the `function_options` given above.

.. note::

    It is also possible to load user-defined representative days, by pointing to a file in `clustering_func` in the same format as pointing to timeseries files in constraints, e.g. ``clustering_func: file=clusters.csv:column_name``. Clusters are unique per datestep, so the clustering file is most readable if the index is at datestep resolution. But, the clustering file index can be in timesteps (e.g. if sharing the same file as a constraint timeseries), with the cluster number repeated per timestep in a day. Cluster values should be integer, starting at zero.

3. Heuristic selection of time steps, that is, the application of one or more of the masks defined in :mod:`calliope.time.masks`, which will mark areas of the time series to retain at maximum resolution (unmasked) and areas where resolution can be lowered (masked). Options can be passed to the masking functions by specifying ``options``. A ``time.function`` can still be specified and will be applied to the masked areas (i.e. those areas of the time series not selected to remain at the maximum resolution), as in this example, which looks for the week of minimum and maximum potential wind generation (assuming a ``wind`` technology was specified), then reduces the rest of the input time series to 6-hourly resolution:

.. code-block:: yaml

    model:
        time:
            masks:
                - {function: extreme, options: {padding: 'calendar_week', tech: 'wind', how: 'max'}}
                - {function: extreme, options: {padding: 'calendar_week', tech: 'wind', how: 'min'}}
            function: resample
            function_options: {'resolution': '6H'}

.. Warning::

  When using time clustering or time masking, the resulting timesteps will be assigned different weights depending on how long a period of time they represent. Weights are used for example to give appropriate weight to the operational costs of aggregated typical days in comparison to individual extreme days, if both exist in the same processed time series. The weighting is accessible in the model data, e.g. through ``Model.inputs.timestep_weights``. The interpretation of results when weights are not 1 for all timesteps requires caution. Production values are not scaled according to weights, but costs are multiplied by weight, in order to weight different timesteps appropriately in the objective function. This means that costs and production values are not consistent without manually post-processing them by either multipyling production by weight (production would then be inconsistent with capacity) or dividing costs by weight. The computation of levelised costs and of capacity factors takes weighting into account, so these values are consisten and can be used as usual.

.. seealso::

  See the implementation of constraints in :mod:`calliope.backend.pyomo.constraints` for more detail on timestep weights and how they affect model constraints.

.. _supply_plus:

The ``supply_plus`` tech
------------------------

The ``plus`` tech groups offer complex functionality, for technologies which cannot be described easily. ``Supply_plus`` allows a supply technology with internal storage of resource before conversion to the carrier happens. This could be emulated with dummy carriers and a combination of supply, storage, and conversion techs, but the ``supply_plus`` tech allows for concise and mathematically more efficient formulation.

.. figure:: images/supply_plus.*
   :alt: supply_plus

   Representation of the ``supply_plus`` technology

An example use of ``supply_plus`` is to define a concentrating solar power (CSP) technology which consumes a solar resource, has built-in thermal storage, and produces electricity. See the :doc:`national-scale built-in example model <tutorials_01_national>` for an application of this.

See the :ref:`listing of supply_plus configuration <abstract_base_tech_definitions>` in the abstract base tech group definitions for the additional constraints that are possible.

.. Warning:: When analysing results from supply_plus, care must be taken to correctly account for the losses along the transformation from resource to carrier. For example, charging of storage from the resource may have a ``resource_eff``-associated loss with it, while discharging storage to produce the carrier may have a different loss resulting from a combination of ``energy_eff`` and ``parasitic_eff``. Such intermediate conversion losses need to be kept in mind when comparing discharge from storage with ``carrier_prod`` in the same time step.

Cyclic storage
--------------

With ``storage`` and ``supply_plus`` techs, it is possible to link the storage at either end of the timeseries, using cyclic storage. This allows the user to better represent multiple years by just modelling one year. Cyclic storage is activated by default (to deactivate: ``run.cyclic_storage: false``). As a result, a technology's initial stored energy at a given location will be equal to its stored energy at the end of the model's last timestep.

For example, for a model running over a full year at hourly resolution, the initial storage at `Jan 1st 00:00:00` will be forced equal to the storage at the end of the timestep `Dec 31st 23:00:00`. By setting ``storage_initial`` for a technology, it is also possible to fix the value in the last timestep. For instance, with ``run.cyclic_storage: true`` and a ``storage_initial`` of zero, the stored energy *must* be zero by the end of the time horizon.

Without cyclic storage in place (as was the case prior to v0.6.2), the storage tech can have any amount of stored energy by the end of the timeseries. This may prove useful in some cases, but has less physical meaning than assuming cyclic storage.

.. note:: Cyclic storage also functions when time clustering, if allowing storage to be tracked between clusters (see :ref:`time_clustering`). However, it cannot be used in ``operate`` run mode.

.. _conversion_plus:

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
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Revenue and export
------------------

It is possible to specify revenues for technologies simply by setting a negative cost value. For example, to consider a feed-in tariff for PV generation, it could be given a negative operational cost equal to the real operational cost minus the level of feed-in tariff received.

Export is an extension of this, allowing an energy carrier to be removed from the system without meeting demand. This is analogous to e.g. domestic PV technologies being able to export excess electricity to the national grid. A cost (or negative cost: revenue) can then be applied to export.

.. note:: Negative costs can be applied to capacity costs, but the user must an ensure a capacity limit has been set. Otherwise, optimisation will be unbounded.

.. _tech_groups:

Using ``tech_groups`` to group configuration
--------------------------------------------

In a large model, several very similar technologies may exist, for example, different kinds of PV technologies with slightly different cost data or with different potentials at different model locations.

To make it easier to specify closely related technologies, ``tech_groups`` can be used to specify configuration shared between multiple technologies. The technologies then give the ``tech_group`` as their parent, rather than one of the abstract base technologies.

For example:

.. code-block:: yaml

    tech_groups:
        pv:
            essentials:
                parent: supply
                carrier: power
            constraints:
                resource: file=pv_resource.csv
                lifetime: 30
            costs:
                monetary:
                    om_annual_investment_fraction: 0.05
                    depreciation_rate: 0.15

    techs:
        pv_large_scale:
            essentials:
                parent: pv
                name: 'Large-scale PV'
            constraints:
                energy_cap_max: 2000
            costs:
                monetary:
                    energy_cap: 750
        pv_rooftop:
            essentials:
                parent: pv
                name: 'Rooftop PV'
            constraints:
                energy_cap_max: 10000
            costs:
                monetary:
                    energy_cap: 1000

None of the ``tech_groups`` appear in model results, they are only used to group model configuration values.

.. _group_share:

Using the ``group_share`` constraint
-------------------------------------

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

.. Note:: The share given in the ``carrier_prod`` constraints refer to the use of generation from ``supply`` and ``supply_plus`` technologies only. The share given in the ``energy_cap`` constraints refers to the combined capacity from ``supply``, ``supply_plus``, ``conversion``, and ``conversion_plus`` technologies.

.. seealso:: The above examples are supplied as overrides in the :ref:`built-in national-scale example <examplemodels_nationalscale_settings>`'s ``scenarios.yaml`` (``cold_fusion`` to define that tech, and ``group_share_cold_fusion_prod`` or ``group_share_cold_fusion_cap`` to apply the group share constraints).

.. _removing_techs_locations:

Removing techs, locations and links
-----------------------------------

By specifying ``exists: false`` in the model configuration, which can be done for example through overrides, model components can be removed for debugging or scenario analysis.

This works for:

* Techs: ``techs.tech_name.exists: false``
* Locations: ``locations.location_name.exists: false``
* Links: ``links.location1,location2.exists: false``
* Techs at a specific location:  ``locations.location_name.techs.tech_name.exists: false``
* Transmission techs at a specific location: ``links.location1,location2.techs.transmission_tech.exists: false``

.. _operational_mode:

Operational mode
----------------

In planning mode, constraints are given as upper and lower boundaries and the model decides on an optimal system configuration. In operational mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm.

To specify a runnable operational model, capacities for all technologies at all locations must have be defined. This can be done by specifying ``energy_cap_equals``. In the absence of ``energy_cap_equals``, constraints given as ``energy_cap_max`` are assumed to be fixed in operational mode.

Operational mode runs a model with a receding horizon control algorithm. This requires two additional settings:

.. code-block:: yaml

    run:
        operation:
            horizon: 48  # hours
            window: 24  # hours

``horizon`` specifies how far into the future the control algorithm optimises in each iteration. ``window`` specifies how many of the hours within ``horizon`` are actually used. In the above example, decisions on how to operate for each 24-hour window are made by optimising over 48-hour horizons (i.e., the second half of each optimisation run is discarded). For this reason, ``horizon`` must always be larger than ``window``.

.. _generating_scripts:

Generating scripts to run a model many times
--------------------------------------------

:ref:`Scenarios and overrides <building_overrides>` can be used to run a given model multiple times with slightly changed settings or constraints.

This functionality can be used together with the ``calliope generate_runs`` and ``calliope generate_scenarios`` command-line tools to generate scripts that run a model many times over in a fully automated way, for example, to explore the effect of different technology costs on model results.

``calliope generate_runs``, at a minimum, must be given the following arguments:

* the model configuration file to use
* the name of the script to create
* ``--kind``: Currently, three options are available. ``windows`` creates a Windows batch (``.bat``) script that runs all models sequentially, ``bash`` creates an equivalent script to run on Linux or macOS, ``bsub`` creates a submission script for a LSF-based high-performance cluster, and ``sbatch`` creates a submission script for a SLURM-based high-performance cluster.
* ``--scenarios``: A semicolon-separated list of scenarios (or overrides/combinations of overrides) to generate scripts for, for example, ``scenario1;scenario2`` or ``override1,override2a;override1,override2b``. Note that when not using manually defined scenario names, a comma is used to group overrides together into a single model -- in the above example, ``override1,override2a`` would be applied to the first run and ``override1,override2b`` be applied to the second run

A fully-formed command generating a Windows batch script to run a model four times with each of the scenarios "run1", "run2", "run3", and "run4":

.. code-block:: shell

    calliope generate_runs model.yaml run_model.bat --kind=windows --scenarios "run1;run2;run3;run4"

Optional arguments are:

* ``--cluster_threads``: specifies the number of threads to request on a HPC cluster
* ``--cluster_mem``: specifies the memory to request on a HPC cluster
* ``--cluster_time``: specifies the run time to request on a HPC cluster
* ``--additional_args``: A text string of any additional arguments to pass directly through to ``calliope run`` in the generated scripts, for example, ``--additional_args="--debug"``.
* ``--debug``: Print additional debug information when running the run generation script.

An example generating a script to run on a ``bsub``-type high-performance cluster, with additional arguments to specify the resources to request from the cluster:

.. code-block:: shell

    calliope generate_runs model.yaml submit_runs.sh --kind=bsub --cluster_mem=1G --cluster_time=100 --cluster_threads=5  --scenarios "run1;run2;run3;run4"

Running this will create two files:

* ``submit_runs.sh``: The cluster submission script to pass to ``bsub`` on the cluster.
* ``submit_runs.array.sh``: The accompanying script defining the runs for the cluster to execute.

In all cases, results are saved into the same directory as the script, with filenames of the form ``out_{run_number}_{scenario_name}.nc`` (model results) and ``plots_{run_number}_{scenario_name}.html`` (HTML plots), where ``{run_number}`` is the run number and ``{scenario_name}`` is the name of the scenario (or the string defining the overrides applied). On a cluster, log files are saved to files with names starting with ``log_`` in the same directory.

Finally, the  ``calliope generate_scenarios`` tool can be used to quickly generate a file with ``scenarios`` definition for inclusion in a model, if a large enough number of overrides exist to make it tedious to manually combine them into scenarios. Assuming that in ``model.yaml`` a range of overrides exist that specify a subset of time for the years 2000 through 2010, called "y2000" through "y2010", and a set of cost-related overrides called "cost_low", "cost_medium" and "cost_high", the following command would generate scenarios with combinations of all years and cost overrides, calling them "run_1", "run_2", and so on, and saving them to ``scenarios.yaml``:

.. code-block:: shell

    calliope generate_scenarios model.yaml scenarios.yaml y2000;y2001;y2002;2003;y2004;y2005;y2006;2007;2008;y2009;2010 cost_low;cost_medium;cost_high --scenario_name_prefix="run_"


.. _imports_in_override_groups:

Imports in overrides
--------------------

When using overrides (see :ref:`building_overrides`), it is possible to have ``import`` statements within overrides for more flexibility. The following example illustrates this:

.. code-block:: yaml

    overrides:
        some_override:
            techs:
                some_tech.constraints.energy_cap_max: 10
            import: [additional_definitions.yaml]

``additional_definitions.yaml``:

.. code-block:: yaml

    techs:
        some_other_tech.constraints.energy_eff: 0.1

This is equivalent to the following override:

.. code-block:: yaml

    overrides:
        some_override:
            techs:
                some_tech.constraints.energy_cap_max: 10
                some_other_tech.constraints.energy_eff: 0.1

Binary and mixed-integer models
-------------------------------

Calliope models are purely linear by default. However, several constraints can turn a model into a binary or mixed-integer model. Because solving problems with binary or integer variables takes considerably longer than solving purely linear models, it usually makes sense to carefully consider whether the research question really necessitates going beyond a purely linear model.

By applying a ``purchase`` cost to a technology, that technology will have a binary variable associated with it, describing whether or not it has been "purchased".

By applying ``units.max``, ``units.min``, or ``units.equals`` to a technology, that technology will have a integer variable associated with it, describing how many of that technology have been "purchased". If a ``purchase`` cost has been applied to this same technology, the purchasing cost will be applied per unit.

.. Warning::

   Integer and binary variables are a recent addition to Calliope and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

.. seealso:: :ref:`milp_example_model`

.. _backend_interface:

Interfacing with the solver backend
-----------------------------------

On loading a model, there is no solver backend, only the input dataset. The backend is generated when a user calls `run()` on their model. Currently this will call back to Pyomo to build the model and send it off to the solver, given by the user in the run configuration ``run.solver``. Once built, solved, and returned, the user has access to the results dataset ``model.results`` and interface functions with the backend ``model.backend``.

You can use this interface to:

1. Get the raw data on the inputs used in the optimisation.
    By running ``model.backend.get_input_params()`` a user get an xarray Dataset which will look very similar to ``model.inputs``, except that assumed default values will be included. You may also spot a bug, where a value in ``model.inputs`` is different to the value returned by this function.

2. Update a parameter value.
    If you are interested in updating a few values in the model, ou can run ``model.backend.update_param()`` . For example, to update your the energy efficiency of your `ccgt` technology in location `region1` from 0.5 to 0.1, you can run ``model.backend.update_param('energy_eff', 'region1::ccgt`, 0.1)``. This will not affect results at this stage, you'll need to rerun the backend (point 4) to optimise with these new values.

3. Activate / Deactivate a constraint or objective.
    Constraints can be activated and deactivate such that they will or will not have an impact on the optimisation. All constraints are active by default, but you might like to remove, for example, a capacity constraint if you don't want there to be a capacity limit for any technologies. Similarly, if you had multiple objectives, you could deactivate one and activate another. The result would be to have a different objective when rerunning the backend.

.. note:: Currently Calliope does not allow you to build multiple objectives, you will need to `understand Pyomo <http://www.pyomo.org/documentation/>`_ and add an additional objective yourself to make use of this functionality. The Pyomo ConcreteModel() object can be accessed at ``model._backend_model``.

4. Rerunning the backend.
    If you have edited parameters or constraint activation, you will need to rerun the optimisation to propagate the effects. By calling ``model.backend.rerun()``, the optimisation will run again, with the updated backend. This will not affect your model, but instead will return a dataset of the inputs/results associated with that *specific* rerun. It is up to you to store this dataset as you see fit. ``model.results`` will remain to be the initial run, and can only be overwritten by ``model.run(force_rerun=True)``.

.. note:: By calling ``model.run(force_rerun=True)`` any updates you have made to the backend will be overwritten.

.. seealso:: :ref:`api_backend_interface`

.. _debugging_runs_config:

Debugging failing runs
----------------------

A Calliope model provides a method to save a fully built and commented model to a single YAML file with ``Model.save_commented_model_yaml(path)``. Comments in the resulting YAML file indicate where values were overridden.

Because this is Calliope's internal representation of a model directly before the ``model_data`` ``xarray.Dataset`` is built, it can be useful for debugging possible issues in the model formulation, for example, undesired constraints that exist at specific locations because they were specified model-wide without having been superseded by location-specific settings.

Two configuration settings can further aid in debugging failing models:

``model.subset_time`` allows specifying a subset of timesteps to be used. This can be useful for debugging purposes as it can dramatically speed up model solution times. The timestep subset can be specified as ``[startdate, enddate]``, e.g. ``['2005-01-01', '2005-01-31']``, or as a single time period, such as ``2005-01`` to select January only. The subsets are processed before building the model and applying time resolution adjustments, so time resolution reduction functions will only see the reduced set of data.

``run.save_logs`` Off by default, if given, sets the directory into which to save logs and temporary files from the backend, to inspect solver logs and solver-generated model files. This also turns on symbolic solver labels in the Pyomo backend, so that all model components in the backend model are named according to the corresponding Calliope model components (by default, Pyomo uses short random names for all generated model components).

.. seealso::

   If using Calliope interactively in a Python session, we recommend reading up on the `Python debugger <https://docs.python.org/3/library/pdb.html>`_ and (if using Jupyter notebooks) making use of the `%debug magic <https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug>`_.

.. _solver_options:

Solver options
--------------

Gurobi
^^^^^^

Refer to the `Gurobi manual <https://www.gurobi.com/documentation/>`_, which contains a list of parameters. Simply use the names given in the documentation (e.g. "NumericFocus" to set the numerical focus value). For example:

.. code-block:: yaml

    run:
        solver: gurobi
        solver_options:
            Threads: 3
            NumericFocus: 2

CPLEX
^^^^^

Refer to the `CPLEX parameter list <https://www.ibm.com/support/knowledgecenter/en/SS9UKU_12.5.0/com.ibm.cplex.zos.help/Parameters/topics/introListAlpha.html>`_. Use the "Interactive" parameter names, replacing any spaces with underscores (for example, the memory reduction switch is called "emphasis memory", and thus becomes "emphasis_memory"). For example:

.. code-block:: yaml

    run:
        solver: cplex
        solver_options:
            mipgap: 0.01
            mip_polishafter_absmipgap: 0.1
            emphasis_mip: 1
            mip_cuts: 2
            mip_cuts_cliques: 3
