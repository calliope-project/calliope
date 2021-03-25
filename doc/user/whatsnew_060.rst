=============
New in v0.6.0
=============

.. Note:: This page summarises changes in version 0.6.0 compared to previous versions. For changes made since version 0.6.0, consult the :doc:`release history <../history>`.

Version `0.6` is backwards incompatible with version `0.5`. If you are familiar with how Calliope functions then this page will act as a reference for moving to version `0.6`.

---------------------
Removed functionality
---------------------

If you require any of the removed functionality, we recommend you `open an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ for it to be built into a later revision of `0.6`.

Technology constraints
======================

* `s_time` (providing a minimum/maximum/exact time of stored energy available for discharge) no longer exists. This constraint was relatively unpredictable in its effects when providing any combination of `s_cap`, `e_cap`, `c_rate` and time clustering.

* The variable `r2` (providing a secondary resource that could be used by a supply/supply_plus technology), along with all its constraints, have been removed. To utilise multiple resource inputs, conversion_plus can be used instead.

* `r_scale_to_peak` (allowing a user to provide a value for the peak resource to which the entire time series would be scaled accordingly) has been removed. `resource_scale` (previously `r_scale`) can still be used for scaling resource values by the given scale factor.

* `weight` (giving a technology a disproportionate weight in the objective function calculation) has been removed.

Custom objectives
=================

The ability to load additional constraints or objectives has been removed. It is still possible to define a custom objective, but to load it, a modeller needs to use a development installation of Calliope and load the function manually.

.. seealso:: :doc:`develop`

---------------------
Updated functionality
---------------------

Verbosity
=========

Almost all sets, constraints, costs, and variables have been updated to be more verbose, making models more readable. The primary updates are:

Sets
----

- `y` -> `techs`
- `x` -> `nodes`
- `c` -> `carriers`
- `k` -> `costs`

Constraints/Costs
-----------------

- `e` -> `energy`, e.g. `e_cap` -> `energy_cap`
- `r` -> `resource`, e.g. `r_cap` -> `resource_cap`
- `s` -> `storage`, e.g. `s_cap` -> `storage_cap`
- `c_rate` -> `charge_rate`
- `p_eff` -> `parasitic_eff`

Variables
---------

- `r` -> `resource_con`: an output from the model giving how much of a resource was consumed
- `r` -> `resource`: the available resource as an input parameter to the model
- `c_prod`/`c_con` -> `carrier_prod`/`carrier_con`: The produced/consumed carrier energy in each time storage_cap

Model and run configuration
===========================

`run.yaml` no longer exists. Instead, all information needed to run a model is now stored in `model.yaml` under the headings `model` and `run`.

`run` *only* contains information about the solver: which one to use and any specific solver options to apply.

`model` contains all other information: time subsetting, model mode, output format, parallel runs, and time clustering.

To solve a model, point to the `model.yaml` file, e.g.: ``calliope run path/to/model.yaml``.

.. _0.6_overrides:

Overrides
=========

Overrides are no longer applied within `run.yaml` (or even `model.yaml`). Instead, overrides are grouped and placed into a separate YAML file, called for example `overrides.yaml` (as of version 0.6.3, they are in a top-level ``overrides`` key in the model configuration, so the informatiom below no longer applies to v0.6.3 and later).

Each group defines any number of overrides to the technology, location, link, model, or run definitions. One or several such groups can then be applied when solving a model, e.g.:

`overrides.yaml`:

.. code-block:: yaml

    higher_costs:
        techs.ccgt.costs.monetary.energy_cap: 10
        locations.region2.techs.csp.costs.monetary.energy_cap: 100
    winter:
        model.subset_time: ['2005-01-01', '2005-02-28']

Running in the command line:

.. code-block:: shell

    calliope run model.yaml --override_file=overrides.yaml:higher_costs

    calliope run model.yaml --override_file=overrides.yaml:higher_costs,winter

Running interactively:

.. code-block:: python

    # only apply the 'higher_costs' override group
    model = calliope.Model(
        'model.yaml',
        override_file='overrides.yaml:higher_costs'
    )

    # apply both the 'higher_costs' and 'winter' overrides
    model2 = calliope.Model(
        'model.yaml',
        override_file='overrides.yaml:higher_costs,winter'
    )

As in version `0.5`, overrides can be applied when creating a `Model` object, via the argument `override_dict`. A dictionary can then be given:

.. code-block:: python

    higher_costs = {
        'techs.ccgt.costs.monetary.energy_cap': 10,
        'locations.region2.techs.csp.costs.monetary.energy_cap': 100
    }

    model = calliope.Model('model.yaml', override_dict=higher_costs)

Parallel runs
=============

Building on the simplified way to define overrides (see above) and on lessons learnt during the development of Calliope so far, the functionality to generate multiple runs to run either on a single machine or in parallel on a high-performance cluster has been greatly simplified and improved.

.. seealso:: :ref:`generating_scripts`

Location and technology subsets
===============================

In model configuration, `subset_x` and `subset_y` (subsetting the used locations and technologies, respectively) no longer exist. `subset_t`, now called `subset_time`, does still exist.

To remove specific technologies or locations from a model, the new and much more powerful ``exists`` option can be used.

.. seealso:: :ref:`removing_techs_locations`

Technology definition
=====================

A technology is now defined in three parts: `essentials`, `constraints`, and `costs`. All top-level definitions (`parent`, `carrier_out`, etc.) are now given under `essentials` and cannot be defined per-location -- they are defined only once for a given technology and apply model-wide. Both `constraints` and `costs` remain the same as in `0.5`, but with more verbose naming:

Old:

.. code-block:: yaml

    supply_grid_power:
        name: 'National grid import'
        parent: supply
        carrier: power
        constraints:
            r: inf
            e_cap.max: 2000
        costs:
            monetary:
                e_cap: 15
                om_fuel: 0.1

New:

.. code-block:: yaml

    supply_grid_power:
        essentials:
            name: 'National grid import'
            parent: supply
            carrier: electricity
        constraints:
            resource: inf
            energy_cap_max: 2000
            lifetime: 25
        costs:
            monetary:
                interest_rate: 0.10
                energy_cap: 15
                om_con: 0.1

Carrier ratios and export carriers have also been moved from essentials into constraints:

Old:

.. code-block:: yaml

    chp:
        name: 'Combined heat and power'
        stack_weight: 100
        parent: conversion_plus
        export: true
        primary_carrier_out: power
        carrier_in: gas
        carrier_out: power
        carrier_out_2:
            heat: 0.8
        constraints:
            e_cap.max: 1500
            e_eff: 0.405
        costs:
            monetary:
                e_cap: 750
                om_var: 0.004
                export: file=export_power.csv

New:

.. code-block:: yaml

    chp:
        essentials:
            name: 'Combined heat and power'
            parent: conversion_plus
            primary_carrier_out: electricity
            carrier_in: gas
            carrier_out: electricity
            carrier_out_2: heat
        constraints:
            export_carrier: electricity
            energy_cap_max: 1500
            energy_eff: 0.405
            carrier_ratios.carrier_out_2.heat: 0.8
            lifetime: 25
        costs:
            monetary:
                interest_rate: 0.10
                energy_cap: 750
                om_prod: 0.004
                export: file=export_power.csv

Per distance constraints and costs have now been incorporated under the constraints and costs keys, with a '_per_distance' suffix:

Old:

.. code-block:: yaml

    heat_pipes:
        name: 'District heat distribution'
        parent: transmission
        carrier: heat
        constraints:
            e_cap.max: 2000
        constraints_per_distance:
            e_loss: 0.025
        costs_per_distance:
            monetary:
                e_cap: 0.3

New:

.. code-block:: yaml

    heat_pipes:
        essentials:
            name: 'District heat distribution'
            parent: transmission
            carrier: heat
        constraints:
            energy_cap_max: 2000
            energy_eff_per_distance: 0.975
            lifetime: 25
        costs:
            monetary:
                interest_rate: 0.10
                energy_cap_per_distance: 0.3

Interest rates and life times
=============================

As seen in the above examples, technology lifetime and interest rate must now be defined for each technology, under `costs`. In version `0.5`, technologies not defining these would silently use implicit default values of 0.10 for interest rate and 25 years for life time. Setting these explicitly for any technology which has investment costs (i.e. those which are not `om_`... or `export`) is now mandatory; no default values exist any more.

Location definition
===================

In version `0.5`, location definitions included a list of technologies to permit at that location(s). An additional `overrides` key permitted per-location changes to model-wide technology definitions.

In `0.6`, "overriding" refers only to model-wide overrides applied :ref:`as described above <0.6_overrides>`. At each location, `techs` simply lists all allowed technologies and any possible changes to model-wide configuration values to apply at this location only, as shown below:

Old:

.. code-block:: yaml

    locations:
        region1:
            techs: [ccgt, csp]
            overrides:
                ccgt:
                    constraints:
                        energy_cap: 100

New:

.. code-block:: yaml

    locations:
        region1:
            techs:
                ccgt:
                    constraints:
                        energy_cap: 100
                # Note that csp must be listed to be permitted here,
                # even though it has no location-specific configuration.
                csp:

Loading time series data from CSV files
=======================================

`x_map` (mapping a technology name to a column in a CSV file) has been removed. Instead, a user can define the time series file column when defining the file name, separated from the file name by a `:`. If no column name is provided, Calliope will look for a column with the location name.

Old:

.. code-block:: yaml

    # will look for the column `demand` in the file `demand_heat_r.csv`
    locations:
        region1:
            techs: [demand_power]
                overrides:
                    demand_power:
                        x_map: demand
                        constraints:
                            r: file

New:

.. code-block:: yaml

    # will look for the column `demand` in the file `demand_heat_r.csv`
    locations:
        region1:
            techs:
                demand_power:
                    constraints:
                        resource: file=demand_heat.csv:demand

Link definition
===============

Links have remained much the same as before. However, there is a slightly different structure in defining technologies, bringing the definition of link technologies more in line with the rest of the model configuration format.

Old:

.. code-block:: yaml

    links:
        region1,region2:
            ac_transmission:
                constraints:
                    e_cap: 1000

New:

.. code-block:: yaml

    links:
        region1,region2:
            techs:
                ac_transmission:
                    constraints:
                        energy_cap: 1000

Location metadata
=================

Location coordinates, previously given under the `metadata` key, are now given directly per location:

Old:

.. code-block:: yaml

    metadata:
        # metadata given in cartesian coordinates, not lat, lon.
        map_boundary:
            lower_left:
                x: 0
                y: 0
            upper_right:
                x: 1
                y: 1
        location_coordinates:
            region1: {x: 2, y: 7}
            region2: {x: 8, y: 7}

New:

.. code-block:: yaml

    locations:
        region1:
            techs:
                ccgt:
                csp:
            coordinates: {x: 2, y: 7}
        region2:
            techs:
                demand_power:
            coordinates: {x: 8, y: 7}


``group_share`` constraint
==========================

The ``group_fraction`` constraint is now called ``group_share`` and has a different formulation more in line with the rest of the tech-specific constraints::

    group_share:
        csp,ccgt:
            energy_cap_min: 0.5
            energy_cap_max: 0.9
            carrier_prod_min:
                power: 0.5

In the process of making these updates, the ``demand_power_peak`` and (undocumented) ``ignored_techs`` options were removed from ``group_share``.

``charge_rate``
===============

When first introduced, charge rate was used to hard-link `energy_cap` and `storage_cap` for a storage/supply_plus technology. This meant that on defining ``energy_cap_max`` and ``charge_rate``, a user was implicitly defining ``storage_cap_max``. This hard-link has now been removed, replaced with only one constraint concerning charge rate: :math:`storage_{cap}(loc::tech) \geq energy_{cap}(loc:tech) \times charge\_rate(loc:tech)`.

.. seealso:: :ref:`constraint_capacity`

Pre-processed data
==================

Version `0.5` kept pre-processed data in either a dictionary (static data), pandas dataframe (location data) or an `xarray Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_ (timeseries data). To view a value that would be used in optimisation, the user would call `model.get_option()`. Similarly, to edit a value before running the model, a user could use `model.set_option()`.

Now, all pre-processed data is held in a single unified `xarray Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_: `model.inputs`.

To view and edit this data before it is sent to the solver, a user need only use standard xarray functionality (see their `documentation <http://xarray.pydata.org/en/stable/>`_ for more information).

Plotting data
=============

.. Note::
    Advanced plotting is still under construction. In case our current functionality is insufficient, input and output data can be plotted by the user using their preferred Python plotting tools, or any other language that can access either NetCDF or CSV data.

Plotting functions can now be called directly on the model and now use `Plotly <https://plot.ly/python/>`_ instead of `0.5`'s matplotlib.

Changes are:

* ``calliope.analysis.plot_capacity(model.solution)`` to ``model.plot.capacity()``

* ``calliope.analysis.plot_transmission(model.solution, carrier='power', tech='ac_transmission')`` to ``model.plot.transmission()``

* ``calliope.analysis.plot_carrier_production(model.solution, carrier='power')`` to ``model.plot.timeseries()``

All available data is plotted, with dropdown menus available for a user to move between plots. A summary of all plotting can also be produced using ``model.plot.summary()``, a function that is also available via the command line interface.

.. seealso:: :ref:`api_model`

Operational mode
================

In `0.6`, running in operational mode changes capacities from decision variables to parameters, preventing various issues that plagued operational mode in prior versions. Additional sense checks were added to ensure that functionality incompatible with operational mode, such as time clustering, is not accidentally used together with it.

.. seealso:: :ref:`operational_mode`

-----------------
New functionality
-----------------

Debugging & checks
==================

A user can now output a data structure of all model input data (the `model_run` dictionary) after Calliope's internal pre-processing, into a YAML file, for debugging. This debug file includes comments as to where constraint/cost values have originated (e.g. having been set by a location-specific configuration, or from a model-wide override group).

Similarly, sense checks are undertaken at several points during pre-processing to ensure the model being built is robust. This includes checks for missing data, possibly misspelled constraints, incompatible inputs, and much more.

This functionality will not find all possible user input errors, as this is an impossible task. However, it flags common mistakes, and the format of implementation allows for further checks to be applied in the future.

Pre-processed model
===================

Having the pre-processed model available in one `xarray Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_ allows a model to be saved to file *before* being run. Although pre-processing is quick, this allows a user to avoid pre-processing the same file multiple times. Instead, they can read in a previously saved NetCDF file which fully describes the model.

Multiple backends
=================

Our primary solver backend is `Pyomo <http://www.pyomo.org/>`_. However, we have now extracted all pre-processing stages from the backend, with all data for a model run being stored in a single `xarray Dataset <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_. This permits the implementation of additional backends.

One such backend currently in an experimental state is based on `JuMP <https://github.com/JuliaOpt/JuMP.jl>`_ in the Julia programming language. Linking Calliope to Julia is a long-term project, for which we welcome any contributions.

Pyomo warmstart
===============

Warmstart functionality can be used in solvers other than GLPK. They allow a previously constructed model to be changed slightly without having to be fully rebuilt. This can speed up re-running a model when you have just a few input parameters you would like to change (the cost of a technology, for instance).

Although the use of warmstart existed in operational mode in version `0.5`, now it extends to all possible parameters in all models. This functionality is currently undocumented in Calliope, but the Pyomo documentation provides some information and the Pyomo model built by Calliope can be accessed by `model._backend_model`.

Backend interface
=================

Once the backend model has been built, it can be accessed by a user, via Calliope. Parameters can be checked and changed, constraints can be activated/deactivated and a model can be re run, all without having to build the backend again. User who are familiar with building large models with Pyomo will be aware of the time penalty associated with processing the model in Pyomo. This additional functionality helps mitigate this, as changing a few parameters need not require complete model rebuild.

.. seealso:: :ref:`api_backend_interface`

Logging
=======

In an interactive Python session (e.g. using Jupyter notebook), output from Calliope can be triggered at different levels of verbosity. By default on building the model (``calliope.Model()``) and running it (``model.run()``), there is no logging displayed unless it is at least a `WARNING`. For helpful information on where the model is in its pre-processing and running in the solver, verbosity can be increased using ``calliope.set_log_level()``.

.. seealso:: :ref:`api_utility_classes`
