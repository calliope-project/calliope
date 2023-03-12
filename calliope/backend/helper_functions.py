import xarray as xr


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


def get_connected_link(model_data, **kwargs):
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

    return _get_connected_link


def get_val_at_index(model_data, **kwargs):
    def _get_val_at_index(*, dim, idx):
        return model_data.coords[dim][int(idx)]

    return _get_val_at_index


def roll(**kwargs):
    def _roll(component, **roll_kwargs):
        roll_kwargs_int = {k: int(v) for k, v in roll_kwargs.items()}
        return component.roll(roll_kwargs_int)

    return _roll
