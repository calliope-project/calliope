# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

import re
from typing import Any, Callable, Union

import xarray as xr

from calliope.exceptions import BackendError


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


def select_from_lookup_arrays(
    model_data: xr.Dataset, as_latex: bool = False, **kwargs
) -> Callable:
    def _as_latex(component: str, **lookup_arrays: str):
        new_strings = {
            (iterator := dim.removesuffix("s")): rf"={array}[{iterator}]"
            for dim, array in lookup_arrays.items()
        }
        component = add_to_iterator(component, new_strings)
        return component

    def _select_from_lookup_arrays(
        component: xr.DataArray, **lookup_arrays: xr.DataArray
    ) -> xr.DataArray:
        """
        Apply vectorised indexing on an arbitrary number of an input array's dimensions.

        Args:
            component (xr.DataArray): Array on which to apply vectorised indexing.

        Kwargs:
            lookup_arrays (dict[str, xr.DataArray]):
                key: dimension on which to apply vectorised indexing
                value: array whose values are either NaN or values from the dimension given in the key.
        Raises:
            BackendError: `component` must be indexed over the dimensions given in the `lookup_arrays` dict keys.
            BackendError: All `lookup_arrays` must be indexed over all the dimensions given in the `lookup_arrays` dict keys.

        Returns:
            xr.DataArray:
                `component` with rearranged values (coordinates remain unchanged).
                Any NaN index coordinates in the lookup arrays will be NaN in the returned array.

        Examples:
        >>> coords = {"foo": ["A", "B", "C"]}
        >>> component = xr.DataArray([1, 2, 3], coords=coords)
        >>> lookup_array = xr.DataArray(
                np.array(["B", "A", np.nan], dtype="O"), coords=coords, name="bar"
            )
        >>> model_data = xr.Dataset({"bar": lookup_array})
        >>> select_from_lookup_arrays(model_data)(component, foo=lookup_array)
        <xarray.DataArray 'bar' (foo: 3)>
        array([ 2.,  1., nan])
        Coordinates:
        * foo      (foo) object 'A' 'B' 'C'

        The lookup array assigns the value at "B" to "A" and vice versa.
        "C" is masked since the lookup array value is NaN.
        """

        dims = set(lookup_arrays.keys())
        missing_dims_in_component = dims.difference(component.dims)
        missing_dims_in_lookup_tables = any(
            dim not in lookup.dims for dim in dims for lookup in lookup_arrays.values()
        )
        if missing_dims_in_component:
            raise BackendError(
                f"Cannot select items from `{component.name}` on the dimensions {dims} since the array is not indexed over the dimensions {missing_dims_in_component}"
            )
        if missing_dims_in_lookup_tables:
            raise BackendError(
                f"All lookup arrays used to select items from `{component.name}` must be indexed over the dimensions {dims}"
            )

        stacked_and_dense_lookup_arrays = {
            # Although we have the lookup array, its values are backend objects,
            # so we grab the same array from the unadulterated model data.
            # FIXME: do not add lookup tables as backend objects.
            dim_name: model_data[lookup.name]
            # Stacking ensures that the dimensions on `component` are not reordered on calling `.sel()`.
            .stack(idx=list(dims))
            # Cannot select on NaNs, so we drop them all.
            .dropna("idx")
            for dim_name, lookup in lookup_arrays.items()
        }
        sliced_component = component.sel(stacked_and_dense_lookup_arrays)

        return (
            sliced_component.drop_vars(dims)
            .unstack("idx")
            .reindex_like(component, copy=False)
        )

    if as_latex:
        return _as_latex
    else:
        return _select_from_lookup_arrays


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
        new_strings = {
            k.removesuffix("s"): f"{-1 * int(v):+d}" for k, v in roll_kwargs.items()
        }
        component = add_to_iterator(component, new_strings)
        return component

    def _roll(component, **roll_kwargs):
        roll_kwargs_int = {k: int(v) for k, v in roll_kwargs.items()}
        return component.roll(roll_kwargs_int)

    if as_latex:
        return _as_latex
    else:
        return _roll


def add_to_iterator(instring: str, iterator_converter: dict[str, str]) -> str:
    """Find an iterator in the iterator substring of the component string
    (anything wrapped in `_text{}`). Other parts of the iterator substring can be anything
    except curly braces, e.g. the standalone `foo` will be found here and acted upon:
    `\\textit{my_param}_\text{bar,foo,foo=bar,foo+1}`

    Args:
        instring (str): String in which the iterator substring can be found.
        iterator_converter (dict[str, str]):
            key: the iterator to search for.
            val: The new string to **append** to the iterator name.

    Returns:
        str: `instring`, but with `iterator` replaced with `iterator + new_string`
    """

    def _replace_in_iterator(matched):
        iterator_list = matched.group(2).split(",")
        new_iterator_list = []
        for it in iterator_list:
            if it in iterator_converter:
                it += iterator_converter[it]
            new_iterator_list.append(it)

        return matched.group(1) + ",".join(new_iterator_list) + matched.group(3)

    return re.sub(r"(_\\text{)([^{}]*?)(})", _replace_in_iterator, instring)
