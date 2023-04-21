=======================
Custom math formulation
=======================

Since Calliope version 0.7, The math used to build optimisation problems is stored in YAML files.

The same syntax used for the `inbuilt math <https://github.com/calliope-project/calliope/tree/main/calliope/math>`_ can be used to define custom math.
So, if you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.

In brief, components of the math formulation are stored under named keys and contain information on the sets over which they will be generated (e.g., for each technology, node, timestep, ...), the conditions under which they will be built in any specific model (e.g., if no `storage` technologies exist in the model, decision variables and constraints associated with them will not be built), and their math expression(s).

In this section, we will describe the math components and the formulation syntax in more detail.
At the end of the section you will find a full reference for the allowed key:value pairs in your custom math YAML file.

.. warning:: When writing custom math, remember that Calliope is a __linear__ modelling framework. It is possible that your desired math will create a nonlinear optimisation problem. Usually, the solver will provide a clear error message when this is the case, although it may not be straightforward to pinpoint what part of your math is the culprit.

.. warning:: Although we have tried to make a generalised syntax for all kinds of custom math, our focus was on reimplementing the base math. Unfortunately, we cannot guarantee that your math will be possible to implement.


-----------
Math syntax
-----------

`foreach` lists
---------------

If the math component is indexed over any Calliope sets (e.g., `techs`, `nodes`, `timesteps`), then you need to define a `foreach` list of those sets. If the decision variable is dimensionless, no `foreach` list needs to be defined.

For example, :yaml:`foreach: [nodes, techs]` will build the component over all `nodes` and `techs` in the model.

The available sets in Calliope are: `nodes`, `techs`, `carriers`, `carrier_tiers`, `costs`, `timesteps`. If using :ref:`time clustering and inter-cluster storage <time_clustering>`, there is also a `datesteps` set available. If you want to build over your own custom set, you will need to add it to the calliope model dataset before building the optimisation problem (:python:`model._model_data.coords[my_new_set] = xr.DataArray(...)`).

`where` strings
---------------
Where strings allow you to define math that applies to only a subset of your data or of the models you are running.
They are made up of a series of statements combined with logical operators.
These statements can be one of the following:

#. Checking the existence of set items in an input parameter. When checking the existence of an input parameter it is possible to first sum it over one or more of its dimensions; if at least one value on the summed dimension(s) is defined, then it will be considered defined in the remaining dimensions. Examples:

        * If you want to apply a constraint across all `nodes` and `techs`, but only for node+tech combinations where the `energy_eff` parameter has been defined, you would include `energy_eff`.
        * If you want to apply a constraint over `techs` and `timesteps`, but only for combinations where the `resource` parameter has at least one `node` where a vaalue is defined, you would include `sum(resource, over=[nodes])`.

#. Checking the value of a configuration option or an input parameter. Checks can use any of the operators: [`>`, `<`, `=`, `<=`, `>=`]. Accessing a value in a set by its position in the index (namely for timeseries data) is possible using `get_val_at_index`. Examples:

    * If you want to apply a constraint only if the configuration option `run.cyclic_storage` is _True_, you would include `run.cyclic_storage=True`. `True`/`False` is case insensitive.
    * If you want to apply a constraint across all `nodes` and `techs`, but only where the `energy_eff` parameter is less than 0.5, you would include `energy_eff<0.5`.
    * If you want to apply a constraint only for the first timestep in your timeseries, you would include `timesteps=get_val_at_index(dim=timesteps, idx=0)`
    * If you want to apply a constraint only for the last timestep in your timeseries, you would include `timesteps=get_val_at_index(dim=timesteps, idx=-1)`

#. Checking the inheritance chain of a technology up to and including the :ref:`abstract_base_tech_definitions`. Examples:

    * If you want to create a decision variable across only `storage` technologies, you would include `inheritance(storage)`.
    * If you want to apply a constraint across only your own `rooftop_supply` technologies and you have assigned the tech_group `rooftop_supply` as the parent of your technologies `pv` and `solar_thermal`, you would include `inheritance(rooftop_supply)`.

#. Subsetting a set. The sets available to subset are always [`nodes`, `techs`, `carriers`, `carrier_tiers`] + any additional sets defined by you in `foreach`. Examples:

    * If you want to create a decision variable for each `carrier` in the model but only if they are _output_ carriers of technologies, you would include `[out, out_2, out_3] in carrier_tiers`.

To combine statements you can use the operators `and`/`or`. You can also use the operator `not` to negate any of the statements. These operators are case-insensitive, so "and", "And", "AND" are equivalent. You can group statements together using the `()` brackets. These statements will be combined first. Examples:

    * If you want to apply a constraint for `storage` technologies if the configuration option `cyclic_storage` is activated and it is the last timestep of the series: `inheritance(storage) and run.cyclic_storage=True and timesteps=get_val_at_index(dim=timesteps, idx=-1)`.
    * If you want to create a decision variable for the input carriers of conversion technologies: `([in] in carrier_tiers and inheritance(conversion))`
    * If you want to apply a constraint if the parameter `resource_unit` is `energy_per_area` or the parameter `resource_area_per_energy_cap` is defined: `resource_unit=energy_per_area or resource_area_per_energy_cap`.
    * If you want to apply a constraint if the parameter `energy_eff` is less than or equal to 0.5 and `resource` has been defined, or `energy_eff` is greater than 0.9 and `resource` has not been defined: `(energy_eff<=0.5 and resource) or (energy_eff>0.9 and not resource)`.

Combining `foreach` and `where` will create an n-dimensional boolean array. Wherever index items in this array are _True_, your component `expression(s)` will be applied.

`expression` strings
--------------------

As with where strings, expression strings are a series of math terms combined with operators. The terms can be input parameters, decision variables, or global expressions.


`equations`
-----------

`sub-expressions`
-----------------

`index-slices`
--------------

---------------
Math components
---------------

Decision variables
------------------
Decision variables (also known as `variables`) are why you are here in the first place.
They are the unknown quantities whose values will decide the value of the objective you are trying to minimise/maximise under the bounds set by the constraints.
These include the output capacity of technologies, the per-timestep flow of carriers into and out of technologies or along transmission lines, and storage content in each timestep.
A decision variable in Calliope math looks like:

.. code-block:: yaml

    variables:
        storage_cap:
            description: "The upper limit on energy that can be stored by a `supply_plus` or `storage` technology in any timestep."
            unit: carrier_unit
            foreach: [nodes, techs]
            where: "(inheritance(storage) OR inheritance(supply_plus)) AND include_storage=True"
            bounds:
                min: storage_cap_min
                max: storage_cap_max
                equals: storage_cap_equals

1. It needs a unique name.
2. Ideally, it has a long-form `description` and a `unit` added. These are not required, but are useful metadata for later reference.
3. Only a top-level

Global Expressions
------------------


Constraints
-----------


Objectives
----------
