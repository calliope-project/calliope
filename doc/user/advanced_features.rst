-----------------
Advanced features
-----------------

Once you're comfortable with :doc:`building <building>`, :doc:`running <running>`, and :doc:`analysing <analysing>` one of the built-in example models, you may want to explore Calliope's advanced functionality. With these features, you will be able to build and run complex models in no time.

.. _time_resolution_adjust:

Time resolution adjustment
--------------------------

Models have a default timestep length (defined implicitly by the timesteps of the model's time series data). This default resolution can be adjusted over parts of the dataset by specifying time resolution adjustment in the model configuration, for example:

.. code-block:: yaml

    config:
        init:
            time_resample: 6H

In the above example, this would resample all time series data to 6-hourly timesteps.
Any `pandas-compatible rule describing the target resolution <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html>`_ can be used.

.. _time_clustering:

Timeseries clustering
---------------------

By supplying a file linking dates in your model timeseries with representative days, it is possible to cluster your timeseries:

.. code-block:: yaml

    config:
        init:
            time_cluster: cluster_days.csv

When using representative days, a number of additional constraints are added, based on the study undertaken by `Kotzur et al <https://doi.org/10.1016/j.apenergy.2018.01.023>`_.
These constraints require a new decision variable ``storage_inter_cluster``, which tracks storage between all the dates of the original timeseries.
This particular functionality can be enabled by including :yaml:`storage_inter_cluster` in your list of custom math.

We no longer provide the functionality to infer representative days from your timeseries.
Instead, we recommend you use other timeseries processing tools applied to your input CSV data or your build model dataset (`model.inputs`).

.. note::

    Resampling and clustering can be applied together.
    Resampling of your timeseries will take place _before_ clustering.

.. warning::

   When using time clustering, the resulting timesteps will be assigned different weights depending on how long a period of time they represent.
   Weights are used for example to give appropriate weight to the operational costs of aggregated typical days in comparison to individual extreme days, if both exist in the same processed time series.
   The weighting is accessible in the model data, e.g. through :python:`model.inputs.timestep_weights`.
   The interpretation of results when weights are not 1 for all timesteps requires caution.
   Production values are not scaled according to weights, but costs are multiplied by weight, in order to weight different timesteps appropriately in the objective function.
   This means that costs and production values are not consistent without manually post-processing them by either multiplying production by weight (production would then be inconsistent with capacity) or dividing costs by weight.
   The computation of levelised costs and of capacity factors takes weighting into account, so these values are consistent and can be used as usual.

.. _tech_groups:

Using ``tech_groups`` to group configuration
--------------------------------------------

In a large model, several very similar technologies may exist, for example, different kinds of PV technologies with slightly different cost data or with different potentials at different model locations.

To make it easier to specify closely related technologies, ``tech_groups`` can be used to specify configuration shared between multiple technologies. The technologies then give the ``tech_group`` as their parent, rather than one of the abstract base technologies.

You can as well extend abstract base technologies, by adding an attribute that will be in effect for all technologies derived from the base technology. To do so, use the name of the abstract base technology for your group, but omit the parent.

For example:

.. code-block:: yaml

    tech_groups:
        supply:
            constraints:
                monetary:
                    interest_rate: 0.1
        pv:
            essentials:
                parent: supply
                carrier: power
            constraints:
                source_max: file=pv_resource.csv
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
                flow_cap_max: 2000
            costs:
                monetary:
                    flow_cap: 750
        pv_rooftop:
            essentials:
                parent: pv
                name: 'Rooftop PV'
            constraints:
                flow_cap_max: 10000
            costs:
                monetary:
                    flow_cap: 1000

None of the ``tech_groups`` appear in model results, they are only used to group model configuration values.


.. _removing_techs_locations:

Removing techs, locations and links
-----------------------------------

By specifying :yaml:`active: false` in the model configuration, which can be done for example through overrides, model components can be removed for debugging or scenario analysis.

This works for:

* Techs: :yaml:`techs.tech_name.active: false`
* Locations: :yaml:`locations.location_name.active: false`
* Links: :yaml:`links.location1,location2.active: false`
* Techs at a specific location:  :yaml:`locations.location_name.techs.tech_name.active: false`
* Transmission techs at a specific location: :yaml:`links.location1,location2.techs.transmission_tech.active: false`

.. _operational_mode:

Operational mode
----------------

In planning mode, constraints are given as upper and lower boundaries and the model decides on an optimal system configuration. In operational mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm.

To specify a runnable operational model, capacities for all technologies at all locations must be defined. This can be done by specifying ``flow_cap_max``, which will be assumed to be the fixed capacity of the technology in operational mode.

Operational mode runs a model with a receding horizon control algorithm. This requires two additional settings:

.. code-block:: yaml

    config.build:
        operate_horizon: 48  # hours
        operate_window: 24  # hours

``horizon`` specifies how far into the future the control algorithm optimises in each iteration. ``window`` specifies how many of the hours within ``horizon`` are actually used. In the above example, decisions on how to operate for each 24-hour window are made by optimising over 48-hour horizons (i.e., the second half of each optimisation run is discarded). For this reason, ``horizon`` must always be larger than ``window``.

.. _spores_mode:

SPORES mode
-----------
SPORES refers to Spatially-explicit Practically Optimal REsultS. This run mode allows a user to generate any number of alternative results which are within a certain range of the optimal cost. It follows on from previous work in the field of `modelling to generate alternatives` (MGA), with a particular emphasis on alternatives that vary maximally in the spatial dimension. This run mode was developed for and implemented in a `study on the future Italian energy system <https://doi.org/10.1016/j.joule.2020.08.002>`_.
As an example, if you wanted to generate 10 SPORES, all of which are within 10% of the optimal system cost, you would define the following in your `run` configuration:

.. code-block:: yaml

    config.build.mode: spores
    config.solve:
        spores_number: 10  # The number of SPORES to generate
        spores_score_cost_class: spores_score  # The cost class to optimise against when generating SPORES
        spores_slack_cost_group: systemwide_cost_max  # The group constraint name in which the `cost_max` constraint is assigned, for use alongside the slack and cost-optimal cost
    parameters:
        slack:
            data: 0.1  # The fraction above the cost-optimal cost to set the maximum cost during SPORES

You will also need to manually set up some other parts of your model to deal with SPORES:

1. Set up a group constraint that can limit the total cost of your system to the SPORES cost (i.e. optimal + 10%). The initial value being infinite ensures it does not impinge on the initial cost-optimal run; the constraint will be adapted internally to set a new value which corresponds to the optimal cost plus the slack.

.. code-block:: yaml

    group_constraints:
        systemwide_cost_max.cost_max.monetary: .inf

2. Assign a `spores_score` cost to all technologies and locations that you want to limit within the scope of finding alternatives. The `spores_score` is the cost class against which the model optimises in the generation of SPORES: technologies at locations with higher scores will be penalised in the objective function, so are less likely to be chosen. In the National Scale example model, this looks like:

.. code-block:: yaml

    techs.ccgt.costs.spores_score.flow_cap: 0
    techs.ccgt.costs.spores_score.interest_rate: 1
    techs.csp.costs.spores_score.flow_cap: 0
    techs.csp.costs.spores_score.interest_rate: 1
    techs.battery.costs.spores_score.flow_cap: 0
    techs.battery.costs.spores_score.interest_rate: 1
    techs.ac_transmission.costs.spores_score.flow_cap: 0
    techs.ac_transmission.costs.spores_score.interest_rate: 1

.. note:: We use and recommend using 'spores_score' and 'systemwide_cost_max' to define the cost class and group constraint, respectively. However, these are user-defined, allowing you to choose terminology that best fits your use-case.

.. _generating_scripts:

Generating scripts to run a model many times
--------------------------------------------

:ref:`Scenarios and overrides <building_overrides>` can be used to run a given model multiple times with slightly changed settings or constraints.

This functionality can be used together with the :sh:`calliope generate_runs` and :sh:`calliope generate_scenarios` command-line tools to generate scripts that run a model many times over in a fully automated way, for example, to explore the effect of different technology costs on model results.

:sh:`calliope generate_runs`, at a minimum, must be given the following arguments:

* the model configuration file to use
* the name of the script to create
* :sh:`--kind`: Currently, three options are available. ``windows`` creates a Windows batch (``.bat``) script that runs all models sequentially, ``bash`` creates an equivalent script to run on Linux or macOS, ``bsub`` creates a submission script for a LSF-based high-performance cluster, and ``sbatch`` creates a submission script for a SLURM-based high-performance cluster.
* :sh:`--scenarios`: A semicolon-separated list of scenarios (or overrides/combinations of overrides) to generate scripts for, for example, ``scenario1;scenario2`` or ``override1,override2a;override1,override2b``. Note that when not using manually defined scenario names, a comma is used to group overrides together into a single model -- in the above example, ``override1,override2a`` would be applied to the first run and ``override1,override2b`` be applied to the second run

A fully-formed command generating a Windows batch script to run a model four times with each of the scenarios "run1", "run2", "run3", and "run4":

.. code-block:: shell

    calliope generate_runs model.yaml run_model.bat --kind=windows --scenarios "run1;run2;run3;run4"

Optional arguments are:

* :sh:`--cluster_threads`: specifies the number of threads to request on a HPC cluster
* :sh:`--cluster_mem`: specifies the memory to request on a HPC cluster
* :sh:`--cluster_time`: specifies the run time to request on a HPC cluster
* :sh:`--additional_args`: A text string of any additional arguments to pass directly through to :sh:`calliope run` in the generated scripts, for example, :sh:`--additional_args="--debug"`.
* :sh:`--debug`: Print additional debug information when running the run generation script.

An example generating a script to run on a ``bsub``-type high-performance cluster, with additional arguments to specify the resources to request from the cluster:

.. code-block:: shell

    calliope generate_runs model.yaml submit_runs.sh --kind=bsub --cluster_mem=1G --cluster_time=100 --cluster_threads=5  --scenarios "run1;run2;run3;run4"

Running this will create two files:

* ``submit_runs.sh``: The cluster submission script to pass to ``bsub`` on the cluster.
* ``submit_runs.array.sh``: The accompanying script defining the runs for the cluster to execute.

In all cases, results are saved into the same directory as the script, with filenames of the form ``out_{run_number}_{scenario_name}.nc`` (model results) and ``plots_{run_number}_{scenario_name}.html`` (HTML plots), where ``{run_number}`` is the run number and ``{scenario_name}`` is the name of the scenario (or the string defining the overrides applied). On a cluster, log files are saved to files with names starting with ``log_`` in the same directory.

Finally, the :sh:`calliope generate_scenarios` tool can be used to quickly generate a file with ``scenarios`` definition for inclusion in a model, if a large enough number of overrides exist to make it tedious to manually combine them into scenarios. Assuming that in ``model.yaml`` a range of overrides exist that specify a subset of time for the years 2000 through 2010, called "y2000" through "y2010", and a set of cost-related overrides called "cost_low", "cost_medium" and "cost_high", the following command would generate scenarios with combinations of all years and cost overrides, calling them "run_1", "run_2", and so on, and saving them to ``scenarios.yaml``:

.. code-block:: shell

    calliope generate_scenarios model.yaml scenarios.yaml y2000;y2001;y2002;2003;y2004;y2005;y2006;2007;2008;y2009;2010 cost_low;cost_medium;cost_high --scenario_name_prefix="run_"


.. _imports_in_override_groups:

Importing other YAML files in overrides
---------------------------------------

When using overrides (see :ref:`building_overrides`), it is possible to have ``import`` statements within overrides for more flexibility. The following example illustrates this:

.. code-block:: yaml

    overrides:
        some_override:
            techs:
                some_tech.constraints.flow_cap_max: 10
            import: [additional_definitions.yaml]

``additional_definitions.yaml``:

.. code-block:: yaml

    techs:
        some_other_tech.constraints.flow_out_eff: 0.1

This is equivalent to the following override:

.. code-block:: yaml

    overrides:
        some_override:
            techs:
                some_tech.constraints.flow_cap_max: 10
                some_other_tech.constraints.flow_out_eff: 0.1

.. _backend_interface:

Interfacing with the solver backend
-----------------------------------

On loading a model, there is no solver backend, only the input dataset. The backend is generated when a user calls `run()` on their model. Currently this will call back to Pyomo to build the model and send it off to the solver, given by the user in the run configuration :yaml:`config.solve.solver`. Once built, solved, and returned, the user has access to the results dataset :python:`model.results` and interface functions with the backend :python:`model.backend`.

You can use this interface to:

1. Get the raw data on the inputs used in the optimisation.
    By running :python:`model.backend.get_input_params()` a user get an xarray Dataset which will look very similar to :python:`model.inputs`, except that assumed default values will be included. You may also spot a bug, where a value in :python:`model.inputs` is different to the value returned by this function.

2. Update a parameter value.
    If you are interested in updating a few values in the model, you can run :python:`model.backend.update_param()`. For example, to update the energy efficiency of your `ccgt` technology in location `region1` from 0.5 to 0.1, you can run :python:`model.backend.update_param('flow_out_eff', {'region1::ccgt`: 0.1})`. This will not affect results at this stage, you'll need to rerun the backend (point 4) to optimise with these new values.

.. note:: If you are interested in updating the objective function cost class weights, you will need to set 'objective_cost_weights' as the parameter, e.g. :python:`model.backend.update_param('objective_cost_weights', {'monetary': 0.5})`.

3. Activate / Deactivate a constraint or objective.
    Constraints can be activated and deactivate such that they will or will not have an impact on the optimisation. All constraints are active by default, but you might like to remove, for example, a capacity constraint if you don't want there to be a capacity limit for any technologies. Similarly, if you had multiple objectives, you could deactivate one and activate another. The result would be to have a different objective when rerunning the backend.

.. note:: Currently Calliope does not allow you to build multiple objectives, you will need to `understand Pyomo <https://www.pyomo.org/documentation/>`_ and add an additional objective yourself to make use of this functionality. The Pyomo ConcreteModel() object can be accessed at :python:`model._backend_model`.

4. Rerunning the backend.
    If you have edited parameters or constraint activation, you will need to rerun the optimisation to propagate the effects. By calling :python:`model.backend.rerun()`, the optimisation will run again, with the updated backend. This will not affect your model, but instead will return a new calliope Model object associated with that *specific* rerun. You can analyse the results and inputs in this new model, but there is no backend interface available. You'll need to return to the original model to access the backend again, or run the returned model using :python:`new_model.run(force_rerun=True)`. In the original model, :python:`model.results` will not change, and can only be overwritten by :python:`model.run(force_rerun=True)`.

.. note:: By calling :python:`model.run(force_rerun=True)` any updates you have made to the backend will be overwritten.

.. seealso:: :ref:`api_backend_interface`

.. _solver_options:

Specifying custom solver options
--------------------------------

Gurobi
^^^^^^

Refer to the `Gurobi manual <https://www.gurobi.com/documentation/>`_, which contains a list of parameters. Simply use the names given in the documentation (e.g. "NumericFocus" to set the numerical focus value). For example:

.. code-block:: yaml

    config.solve:
        solver: gurobi
        solver_options:
            Threads: 3
            NumericFocus: 2

CPLEX
^^^^^

Refer to the `CPLEX parameter list <https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-list-parameters>`_. Use the "Interactive" parameter names, replacing any spaces with underscores (for example, the memory reduction switch is called "emphasis memory", and thus becomes "emphasis_memory"). For example:

.. code-block:: yaml

    config.solve:
        solver: cplex
        solver_options:
            mipgap: 0.01
            mip_polishafter_absmipgap: 0.1
            emphasis_mip: 1
            mip_cuts: 2
            mip_cuts_cliques: 3
