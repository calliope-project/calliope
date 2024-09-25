# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: calliope_docs_build
#     language: python
#     name: calliope_docs_build
# ---

# %% [markdown]
# # Running the national scale example model
# This notebook will show you how to load, build, solve, and examine the results of the national scale example model.

# %%
import pandas as pd
import plotly.express as px

import calliope

# We increase logging verbosity
calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# ## Load model and examine inputs

# %%
model = calliope.examples.national_scale()

# %% [markdown]
# Model inputs can be viewed at `model.inputs`.
# Variables are indexed over any combination of `techs`, `nodes`, `carriers`, `costs` and `timesteps`.

# %%
model.inputs

# %% [markdown]
# Individual data variables can be accessed easily, `to_series().dropna()` allows us to view the data in a nice tabular format.

# %%
model.inputs.flow_cap_max.to_series().dropna()

# %% [markdown]
# You can apply node/tech/carrier/timesteps only operations, like summing information over timesteps

# %%
model.inputs.sink_use_equals.sum(
    "timesteps", min_count=1, skipna=True
).to_series().dropna()

# %% [markdown]
# ## Build and solve the optimisation problem.
#
# Results are loaded into `model.results`.
# By setting the log verbosity at the start of this tutorial to "INFO", we can see the timing of parts of the run, as well as the solver's log.

# %%
model.build()
model.solve()

# %% [markdown]
# Model results are held in the same structure as model inputs.
# The results consist of the optimal values for all decision variables, including capacities and carrier flow.
# There are also results, like system capacity factor and levelised costs, which are calculated in postprocessing before being added to the results Dataset

# %% [markdown]
# ## Examine results

# %%
model.results

# %% [markdown]
# We can sum electricity output over all locations and turn the result into a pandas DataFrame.
#
# Note: electricity output of transmission technologies (e.g., `region1_to_region2`) is the import of electricity at nodes.

# %%
df_electricity = model.results.flow_out.sel(carriers="power").sum("nodes").to_series()
df_electricity.head()

# %% [markdown]
# We can also view total costs associated with each of our technologies at each of our nodes:

# %%
costs = model.results.cost.to_series().dropna()
costs.head()

# %% [markdown]
# We can also examine levelized costs for each location and technology, which is calculated in a post-processing step:

# %%
lcoes = model.results.systemwide_levelised_cost.to_series().dropna()
lcoes

# %% [markdown]
# ### Visualising results
#
# We can use [plotly](https://plotly.com/) to quickly examine our results.
# These are just some examples of how to visualise Calliope data.

# %%
# We set the color mapping to use in all our plots by extracting the colors defined in the technology definitions of our model.
colors = model.inputs.color.to_series().to_dict()

# %% [markdown]
# #### Plotting flows
# We do this by combinging in- and out-flows and separating demand from other technologies.
# First, we look at the aggregated result across all nodes, then we look at each node separately.

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

print(df_electricity.head())

fig1 = px.bar(
    df_electricity_other,
    x="timesteps",
    y="Flow in/out (kWh)",
    color="techs",
    color_discrete_map=colors,
)
fig1.add_scatter(
    x=df_electricity_demand.timesteps,
    y=-1 * df_electricity_demand["Flow in/out (kWh)"],
    marker_color="black",
    name="demand",
)


# %%
df_electricity = (
    (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
    .sel(carriers="power")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow in/out (kWh)")
    .reset_index()
)
df_electricity_demand = df_electricity[df_electricity.techs == "demand_power"]
df_electricity_other = df_electricity[df_electricity.techs != "demand_power"]

print(df_electricity.head())

node_order = df_electricity.nodes.unique()

fig = px.bar(
    df_electricity_other,
    x="timesteps",
    y="Flow in/out (kWh)",
    facet_row="nodes",
    color="techs",
    category_orders={"nodes": node_order},
    height=1000,
    color_discrete_map=colors,
)

showlegend = True
# we reverse the node order (`[::-1]`) because the rows are numbered from bottom to top.
for idx, node in enumerate(node_order[::-1]):
    demand_ = df_electricity_demand.loc[
        df_electricity_demand.nodes == node, "Flow in/out (kWh)"
    ]
    if not demand_.empty:
        fig.add_scatter(
            x=df_electricity_demand.loc[
                df_electricity_demand.nodes == node, "timesteps"
            ],
            y=-1 * demand_,
            row=idx + 1,
            col="all",
            marker_color="black",
            name="Demand",
            legendgroup="demand",
            showlegend=showlegend,
        )
        showlegend = False
fig.update_yaxes(matches=None)
fig.show()

# %% [markdown]
# #### Plotting capacities
# We can plot capacities without needing to combine arrays.
# We first look at flow capacities, then storage capacities.

# %%
df_capacity = (
    model.results.flow_cap.where(model.results.techs != "demand_power")
    .sel(carriers="power")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow capacity (kW)")
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_capacity,
    x="nodes",
    y="Flow capacity (kW)",
    color="techs",
    color_discrete_map=colors,
)
fig.show()

# %%
df_storage_cap = (
    model.results.storage_cap.to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Storage capacity (kWh)")
    .reset_index()
)

print(df_capacity.head())

fig = px.bar(
    df_storage_cap,
    x="nodes",
    y="Storage capacity (kWh)",
    color="techs",
    color_discrete_map=colors,
)
fig.show()

# %% [markdown]
# ### Spatial plots
# Plotly express is limited in its ability to plot spatially,
# but we can at least plot the connections that exist in our results with capacity information available on hover.

# %%
df_coords = model.inputs[["latitude", "longitude"]].to_dataframe().reset_index()
df_capacity = (
    model.results.flow_cap.where(model.inputs.base_tech == "transmission")
    .sel(carriers="power")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow capacity (kW)")
    .reset_index()
)
df_capacity_coords = pd.merge(df_coords, df_capacity, left_on="nodes", right_on="nodes")

fig = px.line_mapbox(
    df_capacity_coords,
    lat="latitude",
    lon="longitude",
    color="techs",
    hover_data="Flow capacity (kW)",
    zoom=3,
    height=300,
)
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=4,
    mapbox_center_lat=df_coords.latitude.mean(),
    mapbox_center_lon=df_coords.longitude.mean(),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
)

# %% [markdown]
# ---

# %% [markdown]
# See the [Calliope documentation](https://calliope.readthedocs.io/) for more details on setting up and running a Calliope model.
