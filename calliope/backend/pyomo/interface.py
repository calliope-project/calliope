import logging

import xarray as xr
import pyomo.core as po  # pylint: disable=import-error

import calliope
from calliope.backend.pyomo.util import get_var
from calliope.backend import run as backend_run
from calliope.backend.pyomo import model as run_pyomo

from calliope.core.util.dataset import reorganise_xarray_dimensions
from calliope.core.util.logging import log_time
from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.analysis.postprocess import postprocess_model_results

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

    return reorganise_xarray_dimensions(xr.Dataset(all_params))


def update_pyomo_param(backend_model, param, update_dict):
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
            'Parameter `{}` not in the Pyomo Backend. Check that the string '
            'matches the corresponding constraint/cost in the model.inputs '
            'xarray Dataset'.format(param)
        )
    elif not isinstance(getattr(backend_model, param), po.base.param.IndexedParam):
        raise exceptions.ModelError(
            '`{}` not a Parameter in the Pyomo Backend. Sets and decision variables '
            'cannot be updated by the user'.format(param)
        )
    elif not isinstance(update_dict, dict):
        raise TypeError('`update_dict` must be a dictionary')

    else:
        print(
            'Warning: we currently do not check that the updated value is the '
            'correct data type for this Pyomo Parameter, this is your '
            'responsibility to check!'
        )
        getattr(backend_model, param).store_values(update_dict)


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
            'constraint/objective `{}` not in the Pyomo Backend.'.format(constraint)
        )
    elif not isinstance(getattr(backend_model, constraint), po.base.Constraint):
        raise exceptions.ModelError(
            '`{}` not a constraint in the Pyomo Backend.'.format(constraint)
        )
    elif active is True:
        getattr(backend_model, constraint).activate()
    elif active is False:
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
    new_model : calliope.Model
        New calliope model, including both inputs and results, but no backend interface.
    """
    backend_model.__calliope_run_config = AttrDict.from_yaml_string(model_data.attrs['run_config'])

    if backend_model.__calliope_run_config['mode'] != 'plan':
        raise exceptions.ModelError(
            'Cannot rerun the backend in {} run mode. Only `plan` mode is '
            'possible.'.format(backend_model.__calliope_run_config['mode'])
        )

    timings = {}
    log_time(logger, timings, 'model_creation')

    results, backend_model = backend_run.run_plan(
        model_data, timings, run_pyomo,
        build_only=False, backend_rerun=backend_model
    )

    inputs = access_pyomo_model_inputs(backend_model)

    # Add additional post-processed result variables to results
    if results.attrs.get('termination_condition', None) in ['optimal', 'feasible']:
        results = postprocess_model_results(
            results, model_data.reindex(results.coords), timings
        )

    for key, var in results.data_vars.items():
        var.attrs['is_result'] = 1

    for key, var in inputs.data_vars.items():
        var.attrs['is_result'] = 0

    new_model_data = xr.merge((results, inputs))
    new_model_data.attrs.update(model_data.attrs)

    # Only add coordinates from the original model_data that don't already exist
    new_coords = [
        i for i in model_data.coords.keys() if i not in new_model_data.coords.keys()
    ]
    new_model_data = new_model_data.update(model_data[new_coords])

    # Reorganise the coordinates so that model data and new model data share
    # the same order of items in each dimension
    new_model_data = new_model_data.reindex(model_data.coords)

    exceptions.warn(
        'The results of rerunning the backend model are only available within '
        'the Calliope model returned by this function call.'
    )

    new_calliope_model = calliope.Model(config=None, model_data=new_model_data)
    new_calliope_model._timings = timings

    return new_calliope_model


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
