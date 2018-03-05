==============
0.5 -> 0.6
==============

Version `0.6` is backwards incompatible with version `0.5`. If you are familiar with how Calliope functions then this page will act as a reference for moving to using version `0.6`.

-------------------------
Depreciated functionality
-------------------------
If you require any depreciated funcitonality, we recommend you open an issue on GitHub for it to be built into a later version of 0.6

Location and technology subsets
===============================

In model configuration, `subset_x` and `subset_y` (allowing subsetting the used locations and technologies, respecitvely) no longer exist. `subset_t`, now `subset_time`, does still exist.

Technology constraints
======================
* `s_time` (providing a minimum/maximum/exact time of stored energy available for discharge) no longer exists. This constraint was relatively volatile when providing any combination of `s_cap`, `e_cap`, `c_rate` and time clustering.

* The variable `r2` (providing a secondary resource that could be used by a supply/supply_plus technology), along with all its constraints, have been removed. To utilise multiple inputs, conversion_plus can be used instead.

* `r_scale_to_peak` (allowing a user to provide a value for the peak resource to which the entire timeseries would be scaled accordingly) has been removed. `resource_scale` (previously `r_scale`) can still be used for scaling resource values

* `weight` (providing a technology a disproportionate weight in the objective function calculation) has been removed.

---------------------
Updated functionality
---------------------

Verbosity
=========

Almost all sets, constraints, costs, and variables have been updated be more verbose. The primary updates are:

Sets
----
- `y` -> `techs`
- `x` -> `locs`
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
- `r` -> `resource_con`: an output from the model of how much resource was consumed
- `r` -> `resource`: the available resource as an input parameter to the model
- `c_prod`/`c_con` -> `carrier_prod`/`carrier_con`: The produced/consumed carrier energy in each time storage_cap


Model and run configuration
===========================
`run.yaml` no longer exists. Instead, the information is all stored in `model.yaml` under the headings `model` and `run`.

`run` *only* contains information about the solver: which one to use and any specific solver options to apply.
`model` contains all other information: time subsetting, model mode, output format, parallel runs, and time clustering.

To call a model, point to the `model.yaml` file.

Overrides
=========
Overrides are no longer applied within `run.yaml` (or even `model.yaml`). Instead, any overrides are grouped and placed into a seperate YAML file, e.g. `overrides.yaml`. Each group defines any overrides to the technology, location, link, model, or run definitions. Each group can then be called when calling the model, e.g.:

`overrides.yaml`:

.. code-block:: yaml

    update_costs:
        techs.ccgt.costs.monetary.energy_cap: 10
        locations.region2.techs.csp.costs.monetary.energy_cap: 100
    winter:
        model.subset_time: ['2005-01-01', '2005-02-28']

Running interactively:

.. code-block:: python

    model = Calliope.Model('model.yaml', override_file='overrides.yaml:update_costs') # only apply the 'update_costs' override group

    model2 = Calliope.Model('model.yaml', override_file='overrides.yaml:update_costs,winter') # apply both the 'update_costs' and 'winter' override groups

Running in command line:

.. code-block:: shell

    calliope run model.yaml --override_file=overrides.yaml:update_costs

    calliope run model.yaml --override_file=overrides.yaml:update_costs,winter


As in `0.5`, overrides can be applied when calling the model, via the argument `override_dict`. A dictionary can then be given:

.. code-block:: python

    update_costs = dict(
        techs=dict(
            ccgt=dict(
                costs=dict(
                    monetary=dict(
                        energycap=10
                    )
                )
            )
        )
        locations=dict(
            region2=dict(
                csp=dict(
                    costs=dict(
                        monetary=dict(
                            energy_cap=100
                        )
                    )
                )
            )
        )
    )

    # or use the following, which is less verbose!
    update_costs = calliope.AttrDict.yaml_from_string(
        """
        techs.ccgt.costs.monetary.energy_cap: 10
        locations.region2.techs.csp.costs.monetary.energy_cap: 100
        """
    )

    model = Calliope.Model('model.yaml', override_dict=update_costs)

Technology definition
=====================
A technology is now defined in three parts: `essentials`, `constraints`, and `costs`. All top-level definitions (`parent`, `carrier_out`, etc.) are now given under `essentials` and cannot be edited at a local level. `constraints` and `costs` remain the same as in 0.5, except with more verbose naming:

old:

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

new:

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

old:

.. code-block:: yaml

    chp:
        name: 'Combined heat and power'
        stack_weight: 100
        parent: conversion_plus
        export: true
        primary_carrier: power
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

new:

.. code-block:: yaml

    chp:
        essentials:
            name: 'Combined heat and power'
            parent: conversion_plus
            primary_carrier: electricity
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

As seen in both above examples, technology lifetime and interest rate have been defined in the new models. These are required for any technology which has investment costs (i.e. those which are not `om_`... or `export`).

Per distance constraints and costs have now been incorporated under the constraints and costs keys, with a '_per_distance' suffix:

old:

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

new:

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

Location definition
===================
At a location level, technologies are defined as YAML keys, not in a list. They can then apply local level constraints, which supercede the global technology constraints:

old:

.. code-block:: yaml

    locations:
        region1:
            techs: [ccgt, csp]
                overrides:
                    ccgt:
                        constraints:
                            energy_cap: 100

new:

.. code-block:: yaml

    locations:
        region1:
            techs:
                ccgt:
                    constraints:
                        energy_cap: 100
                csp: # note that csp is given as a key, but has no local overrides to apply

`x_map` (mapping a technology name to a column in a timeseries file) has been removed. Instead, a used can define the timeseries file column in the same line as defining the file, following a `:`. If no column is provided, the location name will be assumed:

old:

.. code-block:: yaml

    locations:
        region1:
            techs: [demand_power]
                overrides:
                    demand_power:
                        x_map: demand
                        constraints:
                            r: file # will look for the column `demand` in the file `demand_heat_r.csv`

new:

.. code-block:: yaml

    locations:
        region1:
            techs:
                demand_power:
                    constraints:
                        resource: file=demand_heat.csv:demand # will look for the column `demand` in the file `demand_heat_r.csv`

Link definition
===============
Links have remained much the same as before. However, there is a slightly different structure in defining the technologies:

old:

.. code-block:: yaml

    links:
        region1,region2:
            ac_transmission:
                constraints:
                    e_cap: 1000

new:

.. code-block:: yaml

    links:
        region1,region2:
            techs:
                ac_transmission:
                    constraints:
                        energy_cap: 1000

Location metadata
=================
Location coordinates, previously kept under the `metadata` key, are now given per location:

old:

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

new:


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


Preprocessed data
=================
Version `0.5` kept preprocessed data in either a dictionary (static data), pandas dataframe (location data) or an xarray dataset (timeseries data). To view a value that would be used in optimisation, the user would call `model.get_option()`. Similarly, to edit a value before running the model, a user could use `model.set_option()`.

Now, all preprocessed data is held in one xarray dataset: `model.inputs`. To view and edit this data before it is sent to the solver, a user need only use standard xarray functions (see their `documentation <http://xarray.pydata.org/en/stable/>`_ for more information).

Plotting data
=============
.. Note::
    Advanced plotting is still under construction. All input/output data can be plotted by the user, using their preferred method, in case our current functions are insufficient.

Plotting functions can now be called directly on the model and currently use `Plotly <https://plot.ly/python/>`_ instead of matplotlib.

Changes are:

``calliope.analysis.plot_capacity(model.solution)`` -> ``model.plot('capacity', 'energy_cap')``

``calliope.analysis.plot_transmission(model.solution, carrier='power', tech='ac_transmission')`` -> ``model.plot('transmission', 'carrier_prod')``

``calliope.analysis.plot_carrier_production(model.solution, carrier='power')`` ->
``model.plot('timeseries', 'carrier_prod', sum_dims=['locs'], loc=dict(carriers='power'))``

-----------------
New functionality
-----------------

Debugging & checks
==================
A user can now output a verbose dictionary of all model input data (the `model_run` dictionary) into a YAML file, for debugging. This debug file includes comments as to where constraint/cost values have originated (e.g. from being locally supersceded or from an override group).

Similarly, sense checks are undertaken at points during preprocessing to ensure the model being built is robust. It checks for missing data, possibly misspelled constraints, incompatible inputs, and much more. It will not find all possible user input errors, as this is an impossible task. However, the format of implementation allows for further checks to be applied.

Preprocessed model
==================
Having the preprocessed model available in one xarray Dataset allows a model to be saved to file *before* being run. Although preprocessing is quick, this allows a user to avoid preprocessing the same file multiple times, as they can instead call the saved NetCDF file of the model.

Multiple Backends
=================
Our primary solver backend is `Pyomo <http://www.pyomo.org/>`_. However, we have now extracted preprocessing from the backend, with all necessary data for a model run being stored in one xarray Dataset. As such, other backends could be used in future. One such backend which could be used is `JuMP <https://github.com/JuliaOpt/JuMP.jl>`_ in the Julia programming language. Linking Calliope to Julia is a long-term project, for which we welcome any contributions.

Pyomo warmstart
===============

Warmstart functionality can be used in solvers which are not GLPK. They allow a built model to be changed slightly without having to be rebuilt. This can speed up re-running a model when you have just a few input parameters you would like to change (the cost of a technology, for instance). Although this existed in operational mode in version `0.5`, now it extends to all possible parameters in all models. This functionality is undocumented in Calliope, but the Pyomo documentation provides some information and the Pyomo model can be accessed by `model._backend_model`.


