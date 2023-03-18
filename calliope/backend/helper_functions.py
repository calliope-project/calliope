from typing import Callable

import xarray as xr
from calliope.exceptions import BackendError


def inheritance(model_data, **kwargs):
    def _inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    return _inheritance


def imask_sum(model_data, **kwargs):
    def _imask_sum(component, *, over):
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

    return _imask_sum


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


def select_from_lookup_table(model_data: xr.Dataset, **kwargs) -> Callable:
    def _select_from_lookup_table(
        component: xr.DataArray, **slice_dims
    ) -> xr.DataArray:
        dims = set(slice_dims.keys())
        if dims.difference(component.dims):
            raise BackendError(
                f"Cannot select items from `{component.name}` on the dimensions {dims} since the array is not indexed over the dimensions {dims.difference(component.dims)}"
            )
        if any(
            dim not in dim_slicer.dims
            for dim in dims
            for dim_slicer in slice_dims.values()
        ):
            raise BackendError(
                f"All lookup tables used to select items from `{component.name}` must be indexed over the dimensions {dims}"
            )

        sliced_component = component.sel(
            **{
                dim_name: model_data[dim_slicer.name].stack(idx=dims).dropna("idx")
                for dim_name, dim_slicer in slice_dims.items()
            }
        )

        return (
            sliced_component.drop_vars(dims)
            .unstack("idx")
            .reindex_like(component, copy=False)
        )

    return _select_from_lookup_table


def get_val_at_index(model_data, **kwargs):
    def _get_val_at_index(*, dim, idx):
        return model_data.coords[dim][int(idx)]

    return _get_val_at_index


def roll(**kwargs):
    def _roll(component, **roll_kwargs):
        roll_kwargs_int = {k: int(v) for k, v in roll_kwargs.items()}
        return component.roll(roll_kwargs_int)

    return _roll
