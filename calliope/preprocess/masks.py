import operator
import functools
import re

import xarray as xr
import pandas as pd
from calliope.core.attrdict import AttrDict
from calliope.core.util.dataset import reorganise_xarray_dimensions


def create_mask_ds(model_data, sets):
    """
    Create boolean masks for constraints and decision variables
    from data stored in config/sets.yaml
    """
    # TODO: handle 'switches'
    ds = xr.Dataset()
    run_config = AttrDict.from_yaml_string(model_data.attrs["run_config"])

    for set_name, set_items in sets.items():
        for k, v in set_items.items():
            mask = _mask_foreach(model_data, v.foreach)
            if mask is False:  # i.e. not all of 'foreach' are in model_data
                continue
            _operator = "and_"
            if "mask" in v.keys():
                mask = get_mask(model_data, v.mask, mask, _operator)
                for _k, _v in v.get_key("param_idx", default={}).items():
                    if _k in v.foreach:
                        mask = mask.loc[{_k: [i for i in _v if i in mask[_k]]}]
                    else:
                        mask = mask.loc[{_k: [i for i in _v if i in mask[_k]]}].sum(_k) > 0
            if "switches" in v.keys():
                mask = _switch_mask(model_data, mask, v["switches"], run_config)
            if len(mask.dims) > len(v.foreach):
                mask = mask.sum([i for i in mask.dims if i not in v.foreach]) > 0
            elif len(mask.dims) < len(v.foreach):
                raise ValueError(f'Missing dim(s) in mask {k}')
            if isinstance(mask, xr.DataArray) and mask.sum() != 0:
                if k not in ds.data_vars.keys():
                    ds = ds.merge(mask.to_dataset(name=k))
                else:
                    assert (ds[k] == mask).all().item()
                ds[k].attrs[set_name] = 1
                if set_name == 'variables':
                    ds[k].attrs['domain'] = v.get_key('domain', default='Reals')

    return reorganise_xarray_dimensions(ds)


def _param_mask(model_data, param):
    pd.options.mode.use_inf_as_na = True  # can only catch infs as nans using pandas
    if isinstance(model_data.get(param), xr.DataArray):
        _da = model_data.get(param)
        return _da.where(pd.notnull(_da)).notnull()
    else:
        return False
    pd.options.mode.use_inf_as_na = False


def _switch_mask(model_data, mask, switches, run_config):
    for k, v in switches.items():
        for _k, _v in v.items():
            if k == "run":
                mask = _masking(mask, run_config.get_key(_k) == _v, "and")
            elif k == 'tech':
                mask = _masking(mask, model_data[_k] == _v, "and")

    return mask


def _inheritance_mask(model_data, tech_group):
    return model_data.inheritance.str.endswith(tech_group)


def get_mask(model_data, array, initial_mask=None, initial_operator=None):
    masks = []
    operators = []

    for i in array:
        if isinstance(i, list):
            masks.append(get_mask(model_data, i))
        else:
            if "param(" in i:
                _mask = _param_mask(model_data, re.search(r"\((\w+)\)", i).group(1))
                if i.startswith("not"):
                    _mask = ~_mask
                masks.append(_mask)
            elif "inheritance(" in i:
                _mask = _inheritance_mask(
                    model_data, re.search(r"\((\w+)\)", i).group(1)
                )
                if i.startswith("not"):
                    _mask = ~_mask
                masks.append(_mask)
            elif i in ["or", "and"]:
                operators.append(i)
            else:
                raise KeyError(f"unexpected operator or mask: {i}")
    assert len(masks) - 1 == len(operators)
    mask = masks[0]
    for i in range(len(masks) - 1):
        mask = _masking(mask, masks[i + 1], operators[i])

    if initial_mask is not None and initial_operator is not None:
        mask = _masking(mask, initial_mask, initial_operator)

    return mask


def _masking(curr_mask, new_mask, _operator):
    if "or" in _operator:
        _operator = "or_"
    elif "and" in _operator:
        _operator = "and_"
    else:
        raise ValueError(f"Operator `{_operator}` not recognised")
    mask = functools.reduce(getattr(operator, _operator), [curr_mask, new_mask])
    return mask


def _mask_foreach(model_data, foreach):
    if not all(i in model_data.dims for i in foreach):
        return False
    initial_mask = model_data.carrier.notnull() * model_data.node_tech.notnull()
    reduced_mask = (
        initial_mask.sum([i for i in initial_mask.dims if i not in foreach]) > 0
    )

    mask = functools.reduce(
        operator.and_,
        [
            reduced_mask,
            *[model_data[i].notnull() for i in foreach if i not in reduced_mask.dims],
        ],
    )

    return mask
