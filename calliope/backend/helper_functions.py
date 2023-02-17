def inheritance(model_data, **kwargs):
    def _inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    return _inheritance

def backend_sum(**kwargs):
    def _backend_sum(component, *, over):
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

    return _backend_sum


def squeeze_carriers(model_data, **kwargs):
    def _squeeze_carriers(component, carrier_tier):
        return backend_sum(**kwargs)(component.where(
            model_data.carrier.sel(carrier_tiers=carrier_tier).notnull()
        ), over="carriers")

    return _squeeze_carriers


def squeeze_primary_carriers(model_data, **kwargs):
    def _squeeze_primary_carriers(component, carrier_tier):
        return backend_sum(**kwargs)(component.where(
            getattr(model_data, f"primary_carrier_{carrier_tier[0]}").notnull()
        ), over="carriers")

    return _squeeze_primary_carriers


def get_connected_link(model_data, **kwargs):
    def _get_connected_link(component):
        dims = [i for i in component.dims if i in ["techs", "nodes"]]
        remote_nodes = model_data.link_remote_nodes.stack(idx=dims).dropna("idx")
        remote_techs = model_data.link_remote_techs.stack(idx=dims).dropna("idx")
        remote_component_items = component.sel(techs=remote_techs, nodes=remote_nodes)

        return remote_component_items.drop(["nodes", "techs"]).unstack("idx").reindex_like(component).fillna(component)

    return _get_connected_link


def get_timestep(model_data, **kwargs):
    def _get_timestep(ix):
        return model_data.timesteps[int(ix)]
    return _get_timestep


def roll(**kwargs):
    def _roll(component, **roll_kwargs):
        roll_kwargs_int = {k: int(v) for k, v in roll_kwargs.items()}
        return component.roll(roll_kwargs_int)
    return _roll
