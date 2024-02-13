# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: calliope_docs_build
#     language: python
#     name: calliope_docs_build
# ---

# %% [markdown]
# # Pathway optimisation in Calliope
#
# In this tutorial, we use the national scale example model to solve a pathway optimisation problemodel.

# %%

import calliope
import plotly.express as px

calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# # Model input

# %%
# Initialise with the National Scale example model and the "pathways" scenario
model = calliope.examples.national_scale(scenario="pathways")

# %% [markdown]
# ## Assessing the input data
#
# Pathway analysis requires to us have new timeseries dimensions to track investment decisions across years.
# In this example, we define two dimensions: `investsteps` and `vintagesteps`.
#
# At each `investstep`, investment in new capacity is possible and some previously invested capacity is decommissioned as it has reached the end of its lifetime.
# Investing in new capacity is likely to be necessary because of this end-of-life decommissioning but also because of phase-out decommissioning and operational constraints -
# demand might have increased or emissions caps might be more strict requiring more zero-emissions technologies.
# To verify that capacity in a given `investstep` is sufficient to meet that steps's operational constraints, technology dispatch is optimised in each step.
#
# For example, if you have the investment steps `[2030, 2040, 2050]` you will have three points at which to change technology capacities and three sets of annual dispatch decisions
# (e.g., hourly dispatch over a whole year).
#
# Deployed capacity in every investment year will be a combination of capacities deployed in previous years.
# Each will have a different age and may have different characteristics (maintenance costs, efficiency, etc.).
# We refer to each of these previous capacities as "vintages" and track them using `vintagesteps`.
# The available capacity in any `investstep` is the sum of all historical vintages.
# For example, In 2040, the vintages from 2030 and 2040 are available.
#
# We use `vintagesteps` to track two things:
# 1. when a technology is due for decommissioning (if we invested in 100kW of PV in 2030 and it has a 10-year lifetime, it will not be available in 2040).
# 2. the characteristics of older models of a given technology.
# We don't use this at the moment, except to track the investment costs of vintages - these costs are applied in every `investstep` that the given vintage is still available.
#
# Because we use the `steps` suffix for these dimensions, Calliope will read them in as timestamps.

# %%
model.inputs.investsteps

# %%
model.inputs.vintagesteps

# %% [markdown]
# We can see some of our input data is indexed over these dimensions:

# %%
# Decreasing costs of investing in technologies
model.inputs.cost_flow_cap.to_series().dropna()

# %%
# Forced phase-out of combined-cycle gas turbines
model.inputs.flow_cap_max_systemwide.to_series().dropna()

# %% [markdown]
# ### Initial capacity
#
# Unlike greenfield optimisation ("plan" mode in Calliope), pathway optimisation should be initialised with a certain amount of existing technology capacity:

# %%
model.inputs.initial_flow_cap.to_series().dropna()

# %% [markdown]
# This technology capacity can then be phased out as we step through to the end of our time horizon:

# %%
# This is a fraction of the initial capacity that remains available in investment steps
model.inputs.available_initial_cap.to_series().dropna()

# %% [markdown]
# ### Vintages
#
# Technology vintages have specific characteristics, such as the cost to invest in the model.
# Recently commercialised technologies may see their deployment costs decrease substantially in the first decade or two of mass deployment.
# Even established technologies can reduce in cost over time as the manufacturing facilities and logistical pipelines are constantly optimised.

# %%
model.inputs.cost_flow_cap.to_series().dropna()

# %%
model.inputs.cost_storage_cap.to_series().dropna()

# %% [markdown]
# End-of-life decommissioning is tracked with a matrix similar to [initial capacities](#initial-capacity).
# Fractional availability accounts for technologies whose lifetimes fall in-between two investment steps.

# %%
# Note how vintages are never available in investsteps that are in their _future_.
model.inputs.available_vintages.to_series().dropna().unstack("investsteps")

# %% [markdown]
# ## Building a pathways optimisation problem
#
# We have created a math YAML file with updates to all the pre-defined math to handle the existence of `investsteps` and `vintagesteps`.
# Tracking new capacity in each investment period and linking it to technology vintages requires new variables and constraints

# %% [markdown]
# ### Variables

# %%
# Note the "investsteps" dimension added to this pre-defined variable
model.math["variables"]["flow_cap"]

# %%
# This new variable tracks the amount of each technology vintage that exists
model.math["variables"]["flow_cap_new"]

# %% [markdown]
# ### Constraints

# %%
# All existing constraints have the "investsteps" dimension added.
# This allows the dispatch decisions to be optimised individually for each investment step.
model.math["constraints"]["system_balance"]

# %%
# In each investment period, capacities are a combination of all available vintages.
model.math["constraints"]["flow_cap_bounding"]

# %% [markdown]
# ### Building

# %%
model.inputs.flow_cap_max.to_series().dropna()

# %%
model.build()

# %% [markdown]
# ## Analyse results

# %%
model.solve()

# %%
df_capacity = (
    model.results.flow_cap.where(model.results.techs != "demand_power")
    .sel(carriers="power")
    .sum("nodes")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow capacity (kW)")
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_capacity,
    x="investsteps",
    y="Flow capacity (kW)",
    color="techs",
    color_discrete_map=model.inputs.color.to_series().to_dict(),
)
fig.show()

# %%
df_capacity = (
    model.results.storage_cap.sum("nodes")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Storage capacity (kWh)")
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_capacity,
    x="investsteps",
    y="Storage capacity (kWh)",
    color="techs",
    color_discrete_map=model.inputs.color.to_series().to_dict(),
)
fig.show()

# %%
df_capacity = (
    model.results.flow_cap_new.where(model.results.techs != "demand_power")
    .sel(carriers="power")
    .sum("nodes")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("New flow capacity (kW)")
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_capacity,
    x="vintagesteps",
    y="New flow capacity (kW)",
    color="techs",
    color_discrete_map=model.inputs.color.to_series().to_dict(),
)
fig.show()

# %%
df_outflow = (
    (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
    .sel(carriers="power")
    .sum(["nodes", "timesteps"], min_count=1)
    .to_series()
    .where(lambda x: x > 1)
    .dropna()
    .to_frame("Annual outflow (kWh)")
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_outflow,
    x="investsteps",
    y="Annual outflow (kWh)",
    color="techs",
    color_discrete_map=model.inputs.color.to_series().to_dict(),
)
df_demand = (
    model.results.flow_in.sel(techs="demand_power", carriers="power")
    .sum(["nodes", "timesteps"])
    .to_series()
    .reset_index()
)
fig.add_scatter(
    x=df_demand.investsteps, y=df_demand.flow_in, line={"color": "black"}, name="Demand"
)
fig.show()

# %%
df_electricity = (
    (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
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

invest_order = sorted(df_electricity.investsteps.unique())

fig = px.bar(
    df_electricity_other,
    x="timesteps",
    y="Flow in/out (kWh)",
    facet_row="investsteps",
    color="techs",
    height=1000,
    category_orders={"investsteps": invest_order},
    color_discrete_map=model.inputs.color.to_series().to_dict(),
)

showlegend = True
# we reverse the investment year order (`[::-1]`) because the rows are numbered from bottom to top.
for row, year in enumerate(invest_order[::-1]):
    demand_ = df_electricity_demand.loc[(df_electricity_demand.investsteps == year)]
    fig.add_scatter(
        x=demand_["timesteps"],
        y=-1 * demand_["Flow in/out (kWh)"],
        row=row + 1,
        col="all",
        marker_color="black",
        name="Demand",
        legendgroup="demand",
        showlegend=showlegend,
    )
    showlegend = False
fig.update_yaxes(matches=None)
fig.show()
