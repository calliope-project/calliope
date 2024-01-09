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
# # Running the urban scale example model
# This notebook will show you how to load, build, solve, and examine the results of the urban scale example model.

# %%
import calliope
import pandas as pd
import plotly.express as px

# We increase logging verbosity
calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# ## Load model and examine inputs

# %%
model = calliope.examples.urban_scale()

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
model.inputs.sink_equals.sum("timesteps", min_count=1, skipna=True).to_series().dropna()

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
# We can sum heat output over all locations and turn the result into a pandas DataFrame.
#
# Note: heat output of transmission technologies (e.g., `N1_to_X2`) is the import of heat at nodes.

# %%
df_heat = (
    model.results.flow_out.sel(carriers="heat")
    .sum("nodes", min_count=1, skipna=True)
    .to_series()
    .dropna()
    .unstack("techs")
)

df_heat.head()

# %% [markdown]
# We can also examine total technology costs.

# %%
costs = model.results.cost.to_series().dropna()
costs.head()

# %% [markdown]
# We can also examine levelized costs for each location and technology, which is calculated in a post-processing step.

# %%
lcoes = (
    model.results.systemwide_levelised_cost.sel(carriers="electricity")
    .to_series()
    .dropna()
)
lcoes.head()

# %% [markdown]
# ### Visualising results
#
# We can use [plotly](https://plotly.com/) to quickly examine our results.
# These are just some examples of how to visualise Calliope data.

# %%
# We set the color mapping to use in all our plots
colors = model.inputs.color.to_series().to_dict()

# %% [markdown]
# #### Plotting flows
# We do this by combinging in- and out-flows and separating demand from other technologies.
# First, we look at the aggregated result across all nodes for `electricity`, then we look at each node and carrier separately.

# %%
df_electricity = (
    (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
    .sel(carriers="electricity")
    .sum("nodes")
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow in/out (kWh)")
    .reset_index()
)
df_electricity_demand = df_electricity[df_electricity.techs == "demand_electricity"]
df_electricity_other = df_electricity[df_electricity.techs != "demand_electricity"]

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
carriers = ["heat", "electricity"]
df_flows = (
    (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
    .sel(carriers=carriers)
    .to_series()
    .where(lambda x: x != 0)
    .dropna()
    .to_frame("Flow in/out (kWh)")
    .reset_index()
)
df_demand = df_flows[df_flows.techs.str.contains("demand")]
df_flows_other = df_flows[~df_flows.techs.str.contains("demand")]

print(df_flows.head())

node_order = df_flows_other.nodes.unique()

fig = px.bar(
    df_flows_other,
    x="timesteps",
    y="Flow in/out (kWh)",
    facet_row="nodes",
    facet_col="carriers",
    color="techs",
    category_orders={"nodes": node_order, "carriers": carriers},
    height=1000,
    color_discrete_map=colors,
)

showlegend = True
# we reverse the node order (`[::-1]`) because the rows are numbered from bottom to top.
for row, node in enumerate(node_order[::-1]):
    for col, carrier in enumerate(carriers):
        demand_ = df_demand.loc[
            (df_demand.nodes == node) & (df_demand.techs == f"demand_{carrier}"),
            "Flow in/out (kWh)",
        ]
        if not demand_.empty:
            fig.add_scatter(
                x=model.results.timesteps.values,
                y=-1 * demand_,
                row=row + 1,
                col=col + 1,
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
# We can look at capacities for different carriers separately.
# We ignore demand and transmission technology capacities in this example.

# %%
df_capacity = (
    model.results.flow_cap.where(
        ~model.inputs.parent.str.contains("demand|transmission")
    )
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
    facet_col="carriers",
    color_discrete_map=colors,
)
fig.show()

# %% [markdown]
# ### Spatial plots
# Plotly express is limited in its ability to plot spatially,
# but we can at least plot the connections that exist in our results with capacity information available on hover.
# You will only see hover information for one carrier at a time.
# To see the other carrier's information, hide one carrier by clicking on its name in the legend.

# %%
df_coords = model.inputs[["latitude", "longitude"]].to_dataframe().reset_index()
df_capacity = (
    model.results.flow_cap.where(model.inputs.parent == "transmission")
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
    color="carriers",
    hover_name="nodes",
    hover_data="Flow capacity (kW)",
    zoom=3,
    height=300,
)
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=11,
    mapbox_center_lat=df_coords.latitude.mean(),
    mapbox_center_lon=df_coords.longitude.mean(),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    hoverdistance=50,
)

# %% [markdown]
# ---

# %% [markdown]
# See the [Calliope documentation](https://calliope.readthedocs.io/) for more details on setting up and running a Calliope model.
