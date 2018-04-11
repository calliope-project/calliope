import xarray as xr
import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_var
from calliope.backend import run as backend_run
from calliope.backend.pyomo import model as run_pyomo

from calliope.core.util.dataset import reorganise_dataset_dimensions
from calliope.core.util.logging import log_time
from calliope import exceptions


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

    return reorganise_dataset_dimensions(xr.Dataset(all_params))


def update_pyomo_param(backend_model, param, index, value):
    """
    A Pyomo Param value can be updated without the user directly accessing the
    backend model.

    Parameters
    ----------
    param : str
        Name of the parameter to update
    index : tuple of strings
        Tuple of dimension indeces, in the order given in model.inputs for the
        reciprocal parameter
    value : int, float, bool, or str
        Value to assign to the Pyomo Param at the given index

    Returns
    -------
    Value will be updated in-place, requiring the user to run the model again to
    see the effect on results.

    """
    if not hasattr(backend_model, param):
        raise exceptions.ModelError(
            'Parameter {} not in the Pyomo Backend. Check that the string '
            'matches the corresponding constraint/cost in the model.inputs '
            'xarray Dataset'.format(param)
        )
    elif not isinstance(getattr(backend_model, param), po.base.param.IndexedParam):
        raise exceptions.ModelError(
            '{} not a Parameter in the Pyomo Backend. Sets and decision variables '
            'cannot be updated by the user'.format(param)
        )
    elif index not in getattr(backend_model, param):
        raise exceptions.ModelError(
            'index {} not in the Pyomo Parameter {}. call '
            'model.access_backend_model_inputs to see the indeces of the Pyomo '
            'Parameters'.format(index, param)
        )
    else:
        print(
            'Warning: we currently do not check that the updated value is the '
            'correct data type for this Pyomo Parameter, this is your '
            'responsibility to check!'
        )
        getattr(backend_model, param)[index] = value


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
            'constraint/objective {} not in the Pyomo Backend.'.format(constraint)
        )
    elif not isinstance(getattr(backend_model, constraint), po.base.Constraint):
        raise exceptions.ModelError(
            '{} not a constraint in the Pyomo Backend.'.format(constraint)
        )
    elif active:
        getattr(backend_model, constraint).activate()
    elif not active:
        getattr(backend_model, constraint).deactivate()
    else:
        raise ValueError('Argument `active` must be True or False')


def rerun_pyomo_model(model_data, backend_model):
    """
    Rerun the Pyomo backend, perhaps after updating a parameter value,
    (de)activating a constraint/objective or updating run options in the model
    model_data object (e.g. `run.solver`).

    Returns
    -------
    run_data : xarray.Dataset
        Raw data from this rerun, including both inputs and results.
        to filter inputs/results, use `run_data.filter_by_attrs(is_result=...)`
        with 0 for inputs and 1 for results.
    """

    if model_data.attrs['run.mode'] != 'plan':
        raise exceptions.ModelError(
            'Cannot rerun the backend in {} run mode. Only `plan` mode is '
            'possible.'.format(model_data.attrs['run.mode'])
        )

    timings = {}
    log_time(timings, 'model_creation')

    results, backend_model = backend_run.run_plan(
        model_data, timings, run_pyomo,
        build_only=False, backend_rerun=backend_model
    )
    for k, v in timings.items():
        results.attrs['timings.' + k] = v

    exceptions.ModelWarning(
        'model.results will only be updated on running the model from '
        '`model.run()`. We provide results of this rerun as a standalone xarray '
        'Dataset'
    )

    results.attrs.update(model_data.attrs)
    for key, var in results.data_vars.items():
        var.attrs['is_result'] = 1

    inputs = access_pyomo_model_inputs(backend_model)
    for key, var in inputs.data_vars.items():
        var.attrs['is_result'] = 0

    results.update(inputs)
    run_data = results

    return run_data


class BackendInterfaceMethods:

    def __init__(self, model):
        self._backend = model._backend_model
        self._model_data = model._model_data

    def access_model_inputs(self):
        return access_pyomo_model_inputs(self._backend)

    access_model_inputs.__doc__ = access_pyomo_model_inputs.__doc__

    def update_param(self, *args, **kwargs):
        return update_pyomo_param(self._backend, *args, **kwargs)

    update_param.__doc__ = update_pyomo_param.__doc__

    def activate_constraint(self, *args, **kwargs):
        return activate_pyomo_constraint(self._backend, *args, **kwargs)

    activate_constraint.__doc__ = activate_pyomo_constraint.__doc__

    def rerun(self, *args, **kwargs):
        return rerun_pyomo_model(self._model_data, self._backend, *args, **kwargs)

    rerun.__doc__ = rerun_pyomo_model.__doc__
