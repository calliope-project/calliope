# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

import re
from typing import Union, Any

import xarray as xr


def inheritance(model_data, as_latex: bool = False, **kwargs):
    def _as_latex(tech_group):
        return rf"\text{{tech_group={tech_group}}}"

    def _inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    if as_latex:
        return _as_latex
    else:
        return _inheritance


def imask_sum(model_data, as_latex: bool = False, **kwargs):
    def _imask_sum(component, *, over):
        to_return = model_data.get(component, xr.DataArray(False))
        if to_return.any():
            to_return = expression_sum()(to_return, over=over) > 0

        return to_return

    if as_latex:
        return sum_as_latex
    else:
        return _imask_sum


def expression_sum(as_latex: bool = False, **kwargs):
    def _expression_sum(component, *, over):
        """

        Slower method that uses the backend "quicksum" method:

        to_sum_series = to_sum.to_series()
        over = over if isinstance(over, list) else [over]
        summed = backend_interface.sum(to_sum_series, over=over)

        if isinstance(summed, pd.Series):
            to_return = xr.DataArray.from_series(summed)
        else:
            to_return = xr.DataArray(summed)

        Args:
            to_sum (_type_): _description_
            over (_type_): _description_

        Returns:
            _type_: _description_
        """
        to_return = component.sum(over, min_count=1, skipna=True)

        return to_return

    if as_latex:
        return sum_as_latex
    else:
        return _expression_sum


def sum_as_latex(component: str, *, over: Union[str, list]) -> str:
    """Shared utility function to generate the imask and expression latex summation string

    Args:
        component (str): Component name to sum.
        over (Union[str, list]): either one dimension or list of dimension names over which to sum. Each is expected to end in "s".

    Returns:
        str: Valid LaTeX math summation string.
    """

    def _instr(dim):
        dim_singular = dim.removesuffix("s")
        return rf"\text{{{dim_singular}}} \in \text{{{dim}}}"

    if isinstance(over, str):
        overstring = _instr(over)
    else:
        foreach_string = r" \\ ".join(_instr(i) for i in over)
        overstring = rf"\substack{{{foreach_string}}}"
    return rf"\sum\limits_{{{overstring}}} ({component})"


def squeeze_carriers(model_data, as_latex: bool = False, **kwargs):
    def _as_latex(component, carrier_tier):
        return rf"\sum\limits_{{\text{{carrier}} \in \text{{carrier_tier({carrier_tier})}}}} ({component})"

    def _squeeze_carriers(component, carrier_tier):
        return expression_sum(**kwargs)(
            component.where(
                model_data.carrier.sel(carrier_tiers=carrier_tier).notnull()
            ),
            over="carriers",
        )

    if as_latex:
        return _as_latex
    else:
        return _squeeze_carriers


def squeeze_primary_carriers(model_data, as_latex: bool = False, **kwargs):
    def _as_latex(component, carrier_tier):
        return rf"\sum\limits_{{\text{{carrier=primary_carrier_{carrier_tier}}}}} ({component})"

    def _squeeze_primary_carriers(component, carrier_tier):
        return expression_sum(**kwargs)(
            component.where(
                getattr(model_data, f"primary_carrier_{carrier_tier}").notnull()
            ),
            over="carriers",
        )

    if as_latex:
        return _as_latex
    else:
        return _squeeze_primary_carriers


def get_connected_link(model_data, as_latex: bool = False, **kwargs):
    def _as_latex(component):
        for iterator in ["node", "tech"]:
            component = add_to_iterator(component, iterator, f"=remote_{iterator}")
        return component

    def _get_connected_link(component):
        dims = [i for i in component.dims if i in ["techs", "nodes"]]
        remote_nodes = model_data.link_remote_nodes.stack(idx=dims).dropna("idx")
        remote_techs = model_data.link_remote_techs.stack(idx=dims).dropna("idx")
        remote_component_items = component.sel(techs=remote_techs, nodes=remote_nodes)
        return (
            remote_component_items.drop_vars(["nodes", "techs"])
            .unstack("idx")
            .reindex_like(component)
            # TODO: should we be filling NaNs? Should ONLY valid remotes remain in the
            # returned array?
            .fillna(component)
        )

    if as_latex:
        return _as_latex
    else:
        return _get_connected_link


def get_val_at_index(model_data, as_latex: bool = False, **kwargs):
    def _as_latex(*, dim, idx):
        return f"{dim}[{idx}]"

    def _get_val_at_index(*, dim, idx):
        return model_data.coords[dim][int(idx)]

    if as_latex:
        return _as_latex
    else:
        return _get_val_at_index


def roll(as_latex: bool = False, **kwargs):
    def _as_latex(component, **roll_kwargs):
        for k, v in roll_kwargs.items():
            k_singular = k.removesuffix("s")
            component = add_to_iterator(component, k_singular, f"{-1 * int(v):+d}")
        return component

    def _roll(component, **roll_kwargs):
        roll_kwargs_int = {k: int(v) for k, v in roll_kwargs.items()}
        return component.roll(roll_kwargs_int)

    if as_latex:
        return _as_latex
    else:
        return _roll


def add_to_iterator(instring: str, iterator: Any, new_string: Any) -> str:
    """Find an iterator in the iterator substring of the component string
    (anything wrapped in `_text{}`). Other parts of the iterator substring can be anything
    except curly braces, e.g. the standalone `foo` will be found here and acted upon:
    `\\textit{my_param}_\text{bar,foo,foo=bar,foo+1}`

    Args:
        instring (str): String in which the iterator substring can be found.
        iterator (Any): The iterator to search for.
        new_string (Any): The new string to **append** to the iterator name.

    Returns:
        str: `instring`, but with `iterator` replaced with `iterator + new_string`
    """

    def _replace_in_iterator(matched):
        iterator_list = matched.group(2).split(",")
        new_iterator_list = []
        for it in iterator_list:
            if it == iterator:
                it += new_string
            new_iterator_list.append(it)

        return matched.group(1) + ",".join(new_iterator_list) + matched.group(3)

    return re.sub(r"(_\\text{)([^{}]*?)(})", _replace_in_iterator, instring)
