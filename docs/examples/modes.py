# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: calliope_docs_build
#     language: python
#     name: calliope_docs_build
# ---

# %% [markdown]
# # Running models in different modes
#
# Models can be built and solved in different modes:

# - `plan` mode.
# In `plan` mode, the user defines upper and lower boundaries for technology capacities and the model decides on an optimal system configuration.
# In this configuration, the total cost of investing in technologies and then using them to meet demand in every _timestep_ (e.g., every hour) is as low as possible.
# - `operate` mode.
# In `operate` mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm.
# This is sometimes known as a `dispatch` model - we're only concerned with the _dispatch_ of technologies whose capacities are already fixed.
# Optimisation is limited to a time horizon which
# - `spores` mode.
# `SPORES` refers to Spatially-explicit Practically Optimal REsultS.
# This run mode allows a user to generate any number of alternative results which are within a certain range of the optimal cost.

# In this notebook we will run the Calliope national scale example model in these three modes.

# More detail on these modes is given in the [_advanced_ section of the Calliope documentation](https://calliope.readthedocs.io/en/latest/advanced/mode/).

# %%

import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

import calliope

# We update logging to show a bit more information but to hide the solver output, which can be long.
calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# ## Running in `plan` mode.

# %%
# We subset to the same time range as operate mode
model_plan = calliope.examples.national_scale(time_subset=["2005-01-01", "2005-01-10"])
model_plan.build()
model_plan.solve()

# %% [markdown]
# ## Running in `operate` mode.

# %%
model_operate = calliope.examples.national_scale(scenario="operate")
model_operate.build()
model_operate.solve()

# %% [markdown]
# Note how we have capacity variables as parameters in the inputs and only dispatch variables in the results

# %%
model_operate.inputs[["flow_cap", "storage_cap", "area_use"]]

# %%
model_operate.results

# %% [markdown]
# ## Running in `spores` mode.

# %%
# We subset to the same time range as operate/plan mode
model_spores = calliope.examples.national_scale(
    scenario="spores", time_subset=["2005-01-01", "2005-01-10"]
)
model_spores.build()
model_spores.solve()

# %% [markdown]
# Note how we have a new `spores` dimension in our results.

# %%
model_spores.results

# %% [markdown]
# We can track the SPORES scores used between iterations using the `spores_score_cumulative` result.
# This scoring mechanism is based on increasing the score of any technology-node combination where the

# %%
# We do some prettification of the outputs
model_spores.results.spores_score_cumulative.to_series().where(
    lambda x: x > 0
).dropna().unstack("spores")

# %% [markdown]
# ## Visualising results
#
# We can use [plotly](https://plotly.com/) to quickly examine our results.
# These are just some examples of how to visualise Calliope data.

# %%
# We set the color mapping to use in all our plots by extracting the colors defined in the technology definitions of our model.
# We also create some reusable plotting functions.
colors = model_plan.inputs.color.to_series().to_dict()


def plot_flows(results: xr.Dataset) -> go.Figure:
    df_electricity = (
        (results.flow_out.fillna(0) - results.flow_in.fillna(0))
        .sel(carriers="power")
        .sum("nodes")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Flow in/out (kWh)")
        .reset_index()
    )
    df_electricity_demand = df_electricity[df_electricity.techs == "demand_power"]
    df_electricity_other = df_electricity[df_electricity.techs != "demand_power"]

    fig = px.bar(
        df_electricity_other,
        x="timesteps",
        y="Flow in/out (kWh)",
        color="techs",
        color_discrete_map=colors,
    )
    fig.add_scatter(
        x=df_electricity_demand.timesteps,
        y=-1 * df_electricity_demand["Flow in/out (kWh)"],
        marker_color="black",
        name="demand",
    )
    return fig


def plot_capacity(results: xr.Dataset, **plotly_kwargs) -> go.Figure:
    df_capacity = (
        results.flow_cap.where(results.techs != "demand_power")
        .sel(carriers="power")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("Flow capacity (kW)")
        .reset_index()
    )

    fig = px.bar(
        df_capacity,
        x="nodes",
        y="Flow capacity (kW)",
        color="techs",
        color_discrete_map=colors,
        **plotly_kwargs,
    )
    return fig


# %% [markdown]
# ### Using different `spores` scoring algorithms.
#
# We make a number of scoring algorithms accessible out-of-the-box, based on those we present in [Lombardi et al. (2023)](https://doi.org/10.1016/j.apenergy.2023.121002).
# You can call them on solving the model.
# Here, we'll compare the result on `flow_cap` from running each.

# %%
# We subset to the same time range as operate/plan mode
model_spores = calliope.examples.national_scale(
    scenario="spores", time_subset=["2005-01-01", "2005-01-10"]
)
model_spores.build()

spores_results = []
for algorithm in ["integer", "evolving_average", "random", "relative_deployment"]:
    model_spores.solve(**{"spores.scoring_algorithm": algorithm}, force=True)
    spores_results.append(model_spores.results.expand_dims(algorithm=[algorithm]))

spores_results_da = xr.concat(spores_results, dim="algorithm")

spores_results_da.flow_cap.to_series().dropna().unstack("spores")

# %% [markdown]
# ## `plan` vs `operate`
# Here, we compare flows over the 10 days.
# Note how flows do not match as the rolling horizon makes it difficult to make the correct storage charge/discharge decisions.

# %%
fig_flows_plan = plot_flows(
    model_plan.results.sel(timesteps=model_operate.results.timesteps)
)
fig_flows_plan.update_layout(title="Plan mode flows")


# %%
fig_flows_operate = plot_flows(model_operate.results)
fig_flows_operate.update_layout(title="Operate mode flows")

# %% [markdown]
# ## `plan` vs `spores`
# Here, we compare installed capacities between the baseline run (== `plan` mode) and the SPORES.
# Note how the `0` SPORE is the same as `plan` mode and then results deviate considerably.

# %%
fig_flows_plan = plot_capacity(model_plan.results)
fig_flows_plan.update_layout(title="Plan mode capacities")

# %%
fig_flows_spores = plot_capacity(model_spores.results, facet_col="spores")
fig_flows_spores.update_layout(title="SPORES mode capacities")

# %% [markdown]
# ## Comparing `spores` scoring algorithms
# Here, we compare installed capacities between the different SPORES runs.

# %%
fig_flows_spores = plot_capacity(
    spores_results_da, facet_col="spores", facet_row="algorithm"
)
fig_flows_spores.update_layout(
    title="SPORES mode capacities using different scoring algorithms",
    autosize=False,
    height=800,
)
