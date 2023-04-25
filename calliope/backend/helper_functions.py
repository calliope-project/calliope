# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

from typing import Callable

import xarray as xr

from calliope.exceptions import BackendError


def inheritance(model_data, **kwargs):
    def _inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    return _inheritance


def where_sum(model_data, **kwargs):
    def _where_sum(component, *, over):
        """
        Args:
            to_sum (_type_): _description_
            over (_type_): _description_

        Returns:
            _type_: _description_
        """
        to_return = model_data.get(component, xr.DataArray(False))
        if to_return.any():
            to_return = expression_sum()(to_return, over=over) > 0

        return to_return

    return _where_sum


def expression_sum(**kwargs):
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

    return _expression_sum


def squeeze_carriers(model_data, **kwargs):
    def _squeeze_carriers(component, carrier_tier):
        return expression_sum(**kwargs)(
            component.where(
                model_data.carrier.sel(carrier_tiers=carrier_tier).notnull()
            ),
            over="carriers",
        )

    return _squeeze_carriers


def squeeze_primary_carriers(model_data, **kwargs):
    def _squeeze_primary_carriers(component, carrier_tier):
        return expression_sum(**kwargs)(
            component.where(
                getattr(model_data, f"primary_carrier_{carrier_tier}").notnull()
            ),
            over="carriers",
        )

    return _squeeze_primary_carriers


def select_from_lookup_arrays(model_data: xr.Dataset, **kwargs) -> Callable:
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

    return _select_from_lookup_arrays


def get_val_at_index(model_data, **kwargs):
    def _get_val_at_index(*, dim, idx):
        return model_data.coords[dim][int(idx)]

    return _get_val_at_index


def roll(**kwargs):
    def _roll(component, **roll_kwargs):
        roll_kwargs_int = {k: int(v) for k, v in roll_kwargs.items()}
        return component.roll(roll_kwargs_int)

    return _roll
