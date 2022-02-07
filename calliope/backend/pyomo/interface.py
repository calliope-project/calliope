import logging

import xarray as xr
import pyomo.core as po  # pylint: disable=import-error

import calliope
from calliope.backend.pyomo.util import get_var, string_to_datetime
from calliope.backend import run as backend_run
from calliope.backend.pyomo import model as run_pyomo
import calliope.backend.pyomo.interface as pyomo_interface

from calliope.core.util.dataset import reorganise_xarray_dimensions
from calliope.core.util.logging import log_time
from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.postprocess.results import postprocess_model_results

logger = logging.getLogger(__name__)


def access_pyomo_model_inputs(backend_model):
    """
    If the user wishes to inspect the parameter values used as inputs in the backend
    model, they can access a new Dataset of all the backend model inputs, including
    defaults applied where the user did not specify anything for a loc::tech
    """

    all_params = {
        i.name: get_var(backend_model, i.name, sparse=True)
        for i in backend_model.component_objects()
        if isinstance(i, po.base.param.IndexedParam)
    }
    all_param_ds = reorganise_xarray_dimensions(xr.Dataset(all_params))
    all_param_ds = string_to_datetime(backend_model, all_param_ds)
    for var in all_param_ds.data_vars:
        all_param_ds[var].attrs["is_result"] = 0

    return all_param_ds


def update_pyomo_param(backend_model, opt, param, update_dict):
    """
    A Pyomo Param value can be updated without the user directly accessing the
    backend model.

    Parameters
    ----------
    param : str
        Name of the parameter to update
    update_dict : dict
        keys are parameter indeces (either strings or tuples of strings,
        depending on whether there is one or more than one dimension). Values
        are the new values being assigned to the parameter at the given indeces.

    Returns
    -------
    Value(s) will be updated in-place, requiring the user to run the model again to
    see the effect on results.

    """
    if not hasattr(backend_model, param):
        raise exceptions.ModelError(
            "Parameter `{}` not in the Pyomo Backend. Check that the string "
            "matches the corresponding constraint/cost in the model.inputs "
            "xarray Dataset".format(param)
        )
    elif not isinstance(getattr(backend_model, param), po.base.param.IndexedParam):
        raise exceptions.ModelError(
            "`{}` not a Parameter in the Pyomo Backend. Sets and decision variables "
            "cannot be updated by the user".format(param)
        )
    elif not isinstance(update_dict, dict):
        raise TypeError("`update_dict` must be a dictionary")
    else:
        getattr(backend_model, param).store_values(update_dict)

    if opt is not None and "persistent" in opt.name:
        exceptions.warn(
            "Updating the Pyomo parameter won't affect the optimisation run without also "
            "regenerating the relevant constraints or the objective function (see `regenerate_persistent_solver`)."
        )


def regenerate_persistent_pyomo_solver(backend_model, opt, constraints=None, obj=False):
    """
    Having updated a Pyomo Param or several of them, this function can be used
    to regenerate associated constraints in a persistent solver interface, such
    as "gurobi_persistent", before rerunning the model.
    The entire constraint need not be regenerated, it is possible to only point to
    those indexes whose associated parameters have changed.

    Parameters
    ----------
    constraints : dict of lists or None, default = None
        Names of constraints as keys and list of constraint index items as values,
        e.g. `{"energy_capacity_constraint": ["X1::pv"]}` or `{"cost_constraint": [("monetary", "X1::pv"), ("monetary", "X2::pv")])}`.
        Order of index values can be inferred by inspecting the constraint.
    obj : bool, default = False
        If True, will also regenerate the objective function.
    """
    if opt is None or "persistent" not in opt.name:
        raise exceptions.ModelError(
            "Can only regenerate persistent solvers. No persistent solver object found for this model run."
        )

    if obj is True:
        opt.set_objective(backend_model.obj)
    if constraints is not None:
        for constraint_name, constraint_idx in constraints.items():
            for idx in constraint_idx:
                opt.remove_constraint(getattr(backend_model, constraint_name)[idx])
                opt.add_constraint(getattr(backend_model, constraint_name)[idx])

    return opt


def activate_pyomo_constraint(backend_model, constraint, active=True):
    """
    Takes a constraint or objective name, finds it in the backend model and sets
    its status to either active or deactive.

    Parameters
    ----------
    constraint : str
        Name of the constraint/objective to activate/deactivate
        Built-in constraints include '_constraint'
    active: bool, default=True
        status to set the constraint/objective
    """
    if not hasattr(backend_model, constraint):
        raise exceptions.ModelError(
            "constraint/objective `{}` not in the Pyomo Backend.".format(constraint)
        )
    elif not isinstance(getattr(backend_model, constraint), po.base.Constraint):
        raise exceptions.ModelError(
            "`{}` not a constraint in the Pyomo Backend.".format(constraint)
        )
    elif active is True:
        getattr(backend_model, constraint).activate()
    elif active is False:
        getattr(backend_model, constraint).deactivate()
    else:
        raise ValueError("Argument `active` must be True or False")


def rerun_pyomo_model(model_data, backend_model, opt):
    """
    Rerun the Pyomo backend, perhaps after updating a parameter value,
    (de)activating a constraint/objective or updating run options in the model
    model_data object (e.g. `run.solver`).

    Returns
    -------
    new_model : calliope.Model
        New calliope model, including both inputs and results, but no backend interface.
    """
    backend_model.__calliope_run_config = AttrDict.from_yaml_string(
        model_data.attrs["run_config"]
    )
    timings = {}
    log_time(logger, timings, "model_creation")
    inputs = access_pyomo_model_inputs(backend_model)

    run_mode = backend_model.__calliope_run_config["mode"]
    if run_mode == "plan":
        kwargs = {}
    elif run_mode == "spores":
        kwargs = {"interface": pyomo_interface}
    else:
        raise exceptions.ModelError(
            "Cannot rerun the backend in {} run mode. Only `plan` or `spores` modes are "
            "possible.".format(run_mode)
        )
    run_func = getattr(backend_run, f"run_{run_mode}")
    results, backend_model, opt = run_func(
        model_data=model_data,
        timings=timings,
        backend=run_pyomo,
        build_only=False,
        backend_rerun=backend_model,
        opt=opt,
        **kwargs,
    )

    # Add additional post-processed result variables to results
    if results.attrs.get("termination_condition", None) in ["optimal", "feasible"]:
        results = postprocess_model_results(results, model_data, timings)

    for var in results.data_vars.values():
        var.attrs["is_result"] = 1

    for var in inputs.data_vars.values():
        var.attrs["is_result"] = 0

    new_model_data = xr.merge((results, inputs), compat="override")
    new_model_data.attrs.update(model_data.attrs)
    new_model_data.attrs.update(results.attrs)

    # Only add coordinates from the original model_data that don't already exist
    new_coords = [
        i for i in model_data.coords.keys() if i not in new_model_data.coords.keys()
    ]
    new_model_data = new_model_data.update(model_data[new_coords])

    # Reorganise the coordinates so that model data and new model data share
    # the same order of items in each dimension
    new_model_data = new_model_data.reindex(model_data.coords)

    exceptions.warn(
        "The results of rerunning the backend model are only available within "
        "the Calliope model returned by this function call."
    )

    new_calliope_model = calliope.Model(config=None, model_data=new_model_data)
    new_calliope_model._timings = timings

    return opt, new_calliope_model


def add_pyomo_constraint(
    backend_model, constraint_name, constraint_sets, constraint_rule
):
    """
    Functionality to add a constraint to the Pyomo backend, by pointing it to
    a function which implements the mathematics

    Parameters
    ----------
    constraint_name : str
        Name for the pyomo.Constraint object
    constraint_sets : list of str
        Sets over which to implement the constraint.
        The list of sets in the Pyomo model corresponds to model dimensions and
        can be found in full by calling the backend interface method `get_all_model_attrs`
    constraint_rule : function with args (backend_model, *args)
        Function corresponding to the format expected by Pyomo,
        i.e. arguments include the Pyomo backend model (`backend_model`)
        and pointers to each set in `constraint_sets`.
        Set arguments need to be in the same order as in constraint_sets.
        The convention we follow is to name the index without the trailing `s`
        E.g. if 'timesteps' is in constraint_sets, then 'timestep' should be one of the function arguments;
        long `loc_techs` set names are abbreviated to `loc_tech`.
        To see what parameters and variables you have to work with,
        call the backend interface method `get_all_model_attrs`

    Examples
    --------
    To limit supply technologies' capacity to 90% of their maximum possible capacity,
    you would do the following:

    constraint_name = 'max_capacity_90_constraint'
    constraint_sets = ['loc_techs_supply']

    def max_capacity_90_constraint_rule(backend_model, loc_tech):

        return backend_model.energy_cap[loc_tech] <= (
            backend_model.energy_cap_max[loc_tech] * 0.9
        )

    # Add the constraint
    model.backend.add_constraint(constraint_name, constraint_sets, max_capacity_90_constraint_rule)

    # Rerun the model with new constraint.
    # Note: model.run(force_rerun=True) will *not* work, since the backend model will be rebuilt, killing any changes you've made.
    new_model = model.backend.rerun()

    Note that we like the convention that constraint names end with 'constraint' and
    constraint rules have the same text, with an appended '_rule',
    but you are not required to follow this convention to have a working constraint.
    """

    assert (
        constraint_rule.__code__.co_varnames[0] == "backend_model"
    ), "First argument of constraint function must be 'backend_model'."
    assert constraint_rule.__code__.co_argcount - 1 == len(
        constraint_sets
    ), "Number of constraint arguments must equal number of constraint sets + 1."

    try:
        sets = [getattr(backend_model, i) for i in constraint_sets]
    except AttributeError as e:
        e.args = (e.args[0].replace("'ConcreteModel'", "Pyomo backend model"),)
        raise

    setattr(
        backend_model,
        constraint_name,
        po.Constraint(*sets, **{"rule": constraint_rule}),
    )

    return backend_model


def get_all_pyomo_model_attrs(backend_model):
    """
    Get the name of all sets, parameters, and variables in the generated Pyomo model.

    Returns
    -------
    Dictionary differentiating between variables ('Var'), parameters ('Param'), and sets ('Set').
    variables and parameters are given as a dictionaries of lists, where keys are the item names and
    values are a list of dimensions over which they are indexed. These dimensions correspond to the
    sets.
    """
    # Indexed objected
    objects = {
        objname: {
            i.name: [j.name for j in i.index_set().subsets()]
            if i.name + "_index" == i.index_set().name
            else [i.index_set().name]
            for i in backend_model.component_objects()
            if isinstance(i, getattr(po.base, objname))
        }
        for objname in ["Var", "Param"]
    }
    # Indices
    objects["Set"] = [
        i.name for i in backend_model.component_objects() if isinstance(i, po.base.Set)
    ]

    return objects


class BackendInterfaceMethods:
    def __init__(self, model):
        self._backend = model._backend_model
        self._opt = model._backend_model_opt
        self._model_data = model._model_data

    def access_model_inputs(self):
        return access_pyomo_model_inputs(self._backend)

    access_model_inputs.__doc__ = access_pyomo_model_inputs.__doc__

    def update_param(self, *args, **kwargs):
        return update_pyomo_param(self._backend, self._opt, *args, **kwargs)

    update_param.__doc__ = update_pyomo_param.__doc__

    def activate_constraint(self, *args, **kwargs):
        return activate_pyomo_constraint(self._backend, *args, **kwargs)

    activate_constraint.__doc__ = activate_pyomo_constraint.__doc__

    def rerun(self, *args, **kwargs):
        _opt, new_model = rerun_pyomo_model(
            self._model_data, self._backend, self._opt, *args, **kwargs
        )
        self._opt = _opt
        return new_model

    rerun.__doc__ = rerun_pyomo_model.__doc__

    def regenerate_persistent_solver(self, *args, **kwargs):
        self._opt = regenerate_persistent_pyomo_solver(
            self._backend, self._opt, *args, **kwargs
        )

    regenerate_persistent_solver.__doc__ = regenerate_persistent_pyomo_solver.__doc__

    def add_constraint(self, *args, **kwargs):
        self._backend = add_pyomo_constraint(self._backend, *args, **kwargs)

    add_constraint.__doc__ = add_pyomo_constraint.__doc__

    def get_all_model_attrs(self, *args, **kwargs):
        return get_all_pyomo_model_attrs(self._backend, *args, **kwargs)

    get_all_model_attrs.__doc__ = get_all_pyomo_model_attrs.__doc__
