
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
