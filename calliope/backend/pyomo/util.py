from calliope.core.util.tools import memoize


@memoize
def param_getter(backend_model, var, dims):
    try:
        return backend_model.__calliope_model_data__[var][dims]
    except KeyError:  # Try without time dimension, which is always last
        try:
            return backend_model.__calliope_model_data__[var][dims[:-1]]
        except KeyError:  # Static default value
            return backend_model.__calliope_defaults__[var]


def get_previous_timestep(backend_model, timestep):
    # order_dict starts numbering at zero, timesteps is one-indexed, so we do not need
    # to subtract 1 to get to previous_step -- it happens "automagically"
    return backend_model.timesteps[backend_model.timesteps.order_dict[timestep]]
