import xarray as xr
import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_var
from calliope.core.util.dataset import reorganise_dataset_dimensions
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


class BackendInterfaceMethods:

    def __init__(self, model):
        self._backend = model._backend_model

    def access_model_inputs(self):
        return access_pyomo_model_inputs(self._backend)

    access_model_inputs.__doc__ = access_pyomo_model_inputs.__doc__

    def update_param(self, *args, **kwargs):
        return update_pyomo_param(self._backend, *args, **kwargs)

    update_param.__doc__ = update_pyomo_param.__doc__

    def activate_constraint(self, *args, **kwargs):
        return activate_pyomo_constraint(self._backend, *args, **kwargs)

    activate_constraint.__doc__ = activate_pyomo_constraint.__doc__
