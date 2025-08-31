"""Built-in postprocessing functions."""

import xarray as xr

from calliope.postprocess.orchestrator import postprocessor
from calliope.schemas import ModelStructure


def _get_data(name: str, model: ModelStructure) -> xr.DataArray:
    """Get an element from either inputs or results."""
    try:
        data = model.results[name]
    except KeyError:
        data = model.inputs[name]
    return data


def _get_total_generation(model: ModelStructure) -> xr.DataArray:
    """Obtain scaled production by timestep weight."""
    generation = (
        (
            model.results["flow_out"]
            + model.results.get("flow_export", xr.DataArray(0)).fillna(0)
        )
        * model.inputs.timestep_weights
    ).sum(dim=["timesteps", "nodes"], min_count=1)
    return generation


@postprocessor(math=["base"], order=0, active=True)
def capacity_factor(model: ModelStructure) -> xr.DataArray:
    """Calculation of per-technology capacity factors."""
    flow_cap = _get_data("flow_cap", model)

    capacity_factors = (
        model.results["flow_out"]
        / (flow_cap.where(lambda x: x > 0) * model.inputs.timestep_resolution)
    ).where(model.results["flow_out"].notnull())

    return capacity_factors


@postprocessor(math=["base"], order=0, active=True)
def systemwide_capacity_factor(model: ModelStructure) -> xr.DataArray:
    """Calculation of systemwide capacity factors."""
    flow_cap = _get_data("flow_cap", model)

    prod_sum = (model.results["flow_out"] * model.inputs.timestep_weights).sum(
        dim=["timesteps", "nodes"], min_count=1
    )

    cap_sum = flow_cap.where(lambda x: x > 0).sum(dim="nodes", min_count=1)
    time_sum = (model.inputs.timestep_resolution * model.inputs.timestep_weights).sum()

    capacity_factors = (prod_sum / (cap_sum * time_sum)).fillna(0)

    return capacity_factors


@postprocessor(math=["base"], order=0, active=True)
def systemwide_levelised_cost(model: ModelStructure) -> xr.DataArray:
    """Calculates systemwide levelised costs indexed by techs, carriers and costs.

    * the weight of timesteps is accounted for.
    * costs are already multiplied by weight in the constraints, and not further adjusted here.
    * production (`flow_out` + `flow_export`) is not multiplied by weight in the constraints,
      so scaled by weight here to be consistent with costs.
      CAUTION: this scaling is temporary during levelised cost computation -
      the actual costs in the results remain untouched.
    """
    # Here we scale production by timestep weight
    cost = model.results["cost"].sum(dim="nodes", min_count=1)
    generation = _get_total_generation(model)
    levelised_cost = cost / generation.where(lambda x: x > 0)

    return levelised_cost


@postprocessor(math=["base"], order=0, active=True)
def total_levelised_cost(model: ModelStructure) -> xr.DataArray:
    """Calculates total levelised costs by carriers and costs.

    * the weight of timesteps is considered when computing levelised costs:
    * costs are already multiplied by weight in the constraints, and not further adjusted here.
    * production (`flow_out` + `flow_export`) is not multiplied by weight in the constraints,
      so scaled by weight here to be consistent with costs.
      CAUTION: this scaling is temporary during levelised cost computation -
      the actual costs in the results remain untouched.
    """
    cost = model.results["cost"].sum(dim="nodes", min_count=1)
    generation = _get_total_generation(model)

    # `cost` is the total cost of the system
    # `generation`` is only the generation of supply and conversion technologies
    allowed_techs = ("supply", "conversion")
    valid_techs = model.inputs.base_tech.isin(allowed_techs)
    cost = cost.sum(dim="techs", min_count=1)
    generation = generation.sel(techs=valid_techs).sum(dim="techs", min_count=1)

    levelised_cost = cost / generation.where(lambda x: x > 0)
    return levelised_cost


@postprocessor(math=["base"], order=0, active=True)
def unmet_sum(model: ModelStructure) -> xr.DataArray | None:
    """Calculate the sum of unmet demand/supply."""
    if {"unmet_demand", "unused_supply"} & set(model.results.data_vars.keys()):
        unmet_sum = model.results.get("unmet_demand", xr.DataArray(0))
        unmet_sum += model.results.get("unused_supply", xr.DataArray(0))
    else:
        unmet_sum = None
    return unmet_sum
