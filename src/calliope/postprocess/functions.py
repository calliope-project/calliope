"""Built-in postprocessing functions."""

import xarray as xr

from calliope.postprocess.orchestrator import PostprocessContext, postprocessor


def _get_data(name: str, context: PostprocessContext) -> xr.DataArray:
    """Get an element from either inputs or results."""
    if name in context.results.keys():
        data = context.results[name]
    else:
        data = context.inputs[name]
    return data


def _get_total_generation(context: PostprocessContext) -> xr.DataArray:
    """Obtain scaled production by timestep weight."""
    generation = (
        (
            context.results["flow_out"]
            + context.results.get("flow_export", xr.DataArray(0)).fillna(0)
        )
        * context.inputs.timestep_weights
    ).sum(dim=["timesteps", "nodes"], min_count=1)
    return generation


@postprocessor(base_math={"plan"}, order=0, enabled=True)
def capacity_factor(context: PostprocessContext) -> xr.DataArray:
    """Calculation of per-technology capacity factors."""
    flow_cap = _get_data("flow_cap", context)

    capacity_factors = (
        context.results["flow_out"]
        / (flow_cap.where(lambda x: x > 0) * context.inputs.timestep_resolution)
    ).where(context.results["flow_out"].notnull())

    return capacity_factors


@postprocessor(base_math={"plan"}, order=0, enabled=True)
def systemwide_capacity_factor(context: PostprocessContext) -> xr.DataArray:
    """Calculation of systemwide capacity factors."""
    flow_cap = _get_data("flow_cap", context)

    prod_sum = (context.results["flow_out"] * context.inputs.timestep_weights).sum(
        dim=["timesteps", "nodes"], min_count=1
    )

    cap_sum = flow_cap.where(lambda x: x > 0).sum(dim="nodes", min_count=1)
    time_sum = (
        context.inputs.timestep_resolution * context.inputs.timestep_weights
    ).sum()

    capacity_factors = (prod_sum / (cap_sum * time_sum)).fillna(0)

    return capacity_factors


@postprocessor(base_math={"plan"}, order=0, enabled=True)
def systemwide_levelised_cost(context: PostprocessContext) -> xr.DataArray:
    """Calculates systemwide levelised costs indexed by techs, carriers and costs.

    * the weight of timesteps is accounted for.
    * costs are already multiplied by weight in the constraints, and not further adjusted here.
    * production (`flow_out` + `flow_export`) is not multiplied by weight in the constraints,
      so scaled by weight here to be consistent with costs.
      CAUTION: this scaling is temporary during levelised cost computation -
      the actual costs in the results remain untouched.
    """
    # Here we scale production by timestep weight
    cost = context.results["cost"].sum(dim="nodes", min_count=1)
    generation = _get_total_generation(context)
    levelised_cost = cost / generation.where(lambda x: x > 0)

    return levelised_cost


@postprocessor(base_math={"plan"}, order=0, enabled=True)
def total_levelised_cost(context: PostprocessContext) -> xr.DataArray:
    """Calculates total levelised costs by carriers and costs.

    * the weight of timesteps is considered when computing levelised costs:
    * costs are already multiplied by weight in the constraints, and not further adjusted here.
    * production (`flow_out` + `flow_export`) is not multiplied by weight in the constraints,
      so scaled by weight here to be consistent with costs.
      CAUTION: this scaling is temporary during levelised cost computation -
      the actual costs in the results remain untouched.
    """
    cost = context.results["cost"].sum(dim="nodes", min_count=1)
    generation = _get_total_generation(context)

    # `cost` is the total cost of the system
    # `generation`` is only the generation of supply and conversion technologies
    allowed_techs = ("supply", "conversion")
    valid_techs = context.inputs.base_tech.isin(allowed_techs)
    cost = cost.sum(dim="techs", min_count=1)
    generation = generation.sel(techs=valid_techs).sum(dim="techs", min_count=1)

    levelised_cost = cost / generation.where(lambda x: x > 0)
    return levelised_cost


@postprocessor(base_math={"plan"}, order=0, enabled=True)
def unmet_sum(context: PostprocessContext) -> xr.DataArray | None:
    """Calculate the sum of unmet demand/supply."""
    if {"unmet_demand", "unused_supply"} & set(context.results.data_vars.keys()):
        unmet_sum = context.results.get("unmet_demand", 0)
        unmet_sum += context.results.get("unused_supply", 0)
    else:
        unmet_sum = None
    return unmet_sum
