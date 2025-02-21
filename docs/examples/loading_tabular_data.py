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
# # Loading tabular data
#
# In this tutorial, we explore how to load tabular data directly when defining your model,
# instead of / alongside defining the data in YAML.

# %%
from pathlib import Path

import pandas as pd

import calliope
from calliope.io import read_rich_yaml

calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# ## Defining data in the text-based YAML format
#
# The traditional method to define model data in Calliope is to do so in YAML.
# For instance, this simple model contains 2 nodes and a supply, storage, and demand technology at each.
# The nodes are then connected by a transmission technology:
#
# ```yaml
# techs:
#   supply_tech:
#     base_tech: supply
#     carrier_out: electricity
#     flow_cap_max: 10
#     source_use_max:
#       data: [10, 2]
#       index: ["2020-01-01 00:00", "2020-01-01 01:00"]
#       dims: timesteps
#     cost_flow_cap:
#       data: 2
#       index: monetary
#       dims: costs
#
#   storage_tech:
#     base_tech: storage
#     carrier_in: electricity
#     carrier_out: electricity
#     flow_cap_max: 6
#     storage_cap_max: 7
#     cost_storage_cap:
#       data: 5
#       index: monetary
#       dims: costs
#     cost_flow_out:
#       data: 0.1
#       index: monetary
#       dims: costs
#
#   demand_tech:
#     base_tech: demand
#     carrier_in: electricity
#     sink_use_equals:
#       data: [4, 5]
#       index: ["2020-01-01 00:00", "2020-01-01 01:00"]
#       dims: timesteps
#
#   transmission_tech:
#     base_tech: transmission
#     carrier_in: electricity
#     carrier_out: electricity
#     link_from: A
#     link_to: B
#     flow_cap_max: 8
#
# nodes:
#   A.techs: {supply_tech, storage_tech, demand_tech}
#   B.techs:
#     supply_tech:
#       flow_cap_max: 8
#     demand_tech:
# ```
#
# When this is used to initialise a Calliope model, it is processed into a set of data tables ([xarray.DataArray](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html)) internally:

# %%
model_def = read_rich_yaml(
    """
techs:
  supply_tech:
    base_tech: supply
    carrier_out: electricity
    flow_cap_max: 10
    source_use_max:
      data: [10, 2]
      index: ["2020-01-01 00:00", "2020-01-01 01:00"]
      dims: timesteps
    cost_flow_cap:
      data: 2
      index: monetary
      dims: costs

  storage_tech:
    base_tech: storage
    carrier_in: electricity
    carrier_out: electricity
    flow_cap_max: 6
    storage_cap_max: 7
    cost_storage_cap:
      data: 5
      index: monetary
      dims: costs
    cost_flow_out:
      data: 0.1
      index: monetary
      dims: costs

  demand_tech:
    base_tech: demand
    carrier_in: electricity
    sink_use_equals:
      data: [4, 5]
      index: ["2020-01-01 00:00", "2020-01-01 01:00"]
      dims: timesteps

  transmission_tech:
    base_tech: transmission
    carrier_in: electricity
    carrier_out: electricity
    link_from: A
    link_to: B
    flow_cap_max: 8

nodes:
  A.techs: {supply_tech, storage_tech, demand_tech}
  B.techs:
    supply_tech:
      flow_cap_max: 8
    demand_tech:
"""
)
model_from_yaml = calliope.Model(model_def)

# %% [markdown]
# We can look at some of the tabular data we have ended up with.
# Below, we convert the tabular data into [pandas DataFrames](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) since they are very readable.

# %%
model_from_yaml.inputs.source_use_max.to_dataframe()

# %%
model_from_yaml.inputs.flow_cap_max.to_dataframe()

# %%
model_from_yaml.inputs.carrier_in.to_dataframe()

# %% [markdown]
# <div class="admonition note">
#     <p class="admonition-title">Note</p>
#     <p>
#         `carrier_in` and `carrier_out` are provided as strings in YAML but are converted to a binary "lookup" array within Calliope.
#         The carrier names have become entries in the `carriers` dimension.
#     </p>
# </div>

# %%
model_from_yaml.inputs.cost_flow_cap.to_dataframe()

# %% [markdown]
# ## Defining data in the tabular CSV format

# %% [markdown]
# We could have defined these same tables in CSV files and loaded them using `data_tables`.
# We don't yet have those CSV files ready, so we'll create them programmatically.
# In practice, you would likely write these files using software like Excel.

# We do not create one big table for all the data, but instead group data with similar dimensions together.
# Therefore, timeseries data goes in one file, cost data in another, and data linking technologies to nodes or carriers into their own files.

# We also create tables with different shapes.
# Some are long and thin with all the dimensions grouped in each row (or the `index`), while others have dimensions grouped in the columns.
# This is to show what is possible.
# You might choose to always have long and thin data, or to always have certain dimensions in the rows and others in the columns.
# So long as you then define your data table correctly in the model definition, so that Calliope knows exactly how to process your data, it doesn't matter what shape it is stored in.

# %% [markdown]
# First, we create a directory to hold the tabular data we are about to generate.

# %%
data_table_path = Path(".") / "outputs" / "loading_tabular_data"
data_table_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# Next we group together **technology data where no extra dimensions are needed**.
# This means the basics like specifying a `base_tech` for each technology.
# We generate this data as a table and save it to a file called `tech_data.csv`.
# %%
tech_data = pd.DataFrame(
    {
        "supply_tech": {"base_tech": "supply"},
        "storage_tech": {
            "base_tech": "storage",
            "flow_cap_max": 6,
            "storage_cap_max": 7,
        },
        "demand_tech": {"base_tech": "demand"},
        "transmission_tech": {
            "base_tech": "transmission",
            "link_from": "A",
            "link_to": "B",
            "flow_cap_max": 8,
        },
    }
)
tech_data.to_csv(data_table_path / "tech_data.csv")
tech_data

# %% [markdown]
# Now we deal with technology data that **requires the `timesteps` dimension**, again defining it as a table which we save to a CSV file:
# %%
tech_timestep_data = pd.DataFrame(
    {
        ("supply_tech", "source_use_max"): {
            "2020-01-01 00:00": 10,
            "2020-01-01 01:00": 2,
        },
        ("demand_tech", "sink_use_equals"): {
            "2020-01-01 00:00": 4,
            "2020-01-01 01:00": 5,
        },
    }
)
tech_timestep_data.to_csv(data_table_path / "tech_timestep_data.csv")
tech_timestep_data

# %% [markdown]
# The same procedure for **technology data with the `carriers` dimension**:
#
# Note that there are no carriers mentioned in this file.
# Instead, we will add the dimension when we load the file
# (since it is the same value - `electricity` - for all rows).
# %%
tech_carrier_data = pd.Series(
    {
        ("supply_tech", "carrier_out"): 1,
        ("storage_tech", "carrier_in"): 1,
        ("storage_tech", "carrier_out"): 1,
        ("demand_tech", "carrier_in"): 1,
        ("transmission_tech", "carrier_in"): 1,
        ("transmission_tech", "carrier_out"): 1,
    }
)
tech_carrier_data.to_csv(data_table_path / "tech_carrier_data.csv")
tech_carrier_data
# %% [markdown]
# And the **technology data with the `nodes` dimension**:
# %%
tech_node_data = pd.Series(
    {("supply_tech", "B", "flow_cap_max"): 8, ("supply_tech", "A", "flow_cap_max"): 10}
)
tech_node_data.to_csv(data_table_path / "tech_node_data.csv")
tech_node_data
# %% [markdown]
# Finally, we deal with the **technology data with the `costs` dimension**.
#
# As with the `carriers` dimension data above, we do not explicitly define the costs dimension as,
# once again, it is a single value: `monetary`.
# Instead of repeating it multiple times, we just add it on when we load in the file.
# %%
tech_cost_data = pd.DataFrame(
    {
        "storage_tech": {"cost_storage_cap": 5, "cost_flow_out": 0.1},
        "supply_tech": {"cost_flow_cap": 2},
    }
)
tech_cost_data.to_csv(data_table_path / "tech_cost_data.csv")
tech_cost_data

# %% [markdown]
# Now our YAML model definition can simply link to each of the CSV files we created in the `data_tables` section, instead of needing to define the data in YAML directly:
#
# ```yaml
# data_tables:
#   tech_data:
#     data: outputs/loading_tabular_data/tech_data.csv
#     rows: parameters
#     columns: techs
#   tech_node_data:
#     data: outputs/loading_tabular_data/tech_node_data.csv
#     rows: [techs, nodes, parameters]
#   tech_timestep_data:
#     data: outputs/loading_tabular_data/tech_timestep_data.csv
#     rows: timesteps
#     columns: [techs, parameters]
#   tech_carrier_data:
#     data: outputs/loading_tabular_data/tech_carrier_data.csv
#     rows: [techs, parameters]
#     add_dims:
#       carriers: electricity
#   tech_cost_data:
#     data: outputs/loading_tabular_data/tech_cost_data.csv
#     rows: parameters
#     columns: techs
#     add_dims:
#       costs: monetary
# ```
#
# When loading data tables, assigning techs to nodes is done automatically to some extent.
# That is, if a tech is defined at a node in a data table (in this case, only for `supply_tech`), then Calliope assumes that this tech should be allowed to exist at the corresponding node.
# Since it is easy to lose track of which parameters you've defined at nodes and which ones not, it is _much_ safer to explicitly define a list of technologies at each node in your YAML definition:
#
# ```yaml
# nodes:
#   A.techs: {supply_tech, storage_tech, demand_tech}
#   B.techs: {supply_tech, demand_tech}
# ```

# %%
model_def = read_rich_yaml(
    """
data_tables:
  tech_data:
    data: outputs/loading_tabular_data/tech_data.csv
    rows: parameters
    columns: techs
  tech_node_data:
    data: outputs/loading_tabular_data/tech_node_data.csv
    rows: [techs, nodes, parameters]
  tech_timestep_data:
    data: outputs/loading_tabular_data/tech_timestep_data.csv
    rows: timesteps
    columns: [techs, parameters]
  tech_carrier_data:
    data: outputs/loading_tabular_data/tech_carrier_data.csv
    rows: [techs, parameters]
    add_dims:
      carriers: electricity
  tech_cost_data:
    data: outputs/loading_tabular_data/tech_cost_data.csv
    rows: parameters
    columns: techs
    add_dims:
      costs: monetary
nodes:
  A.techs: {supply_tech, storage_tech, demand_tech}
  B.techs: {supply_tech, demand_tech}
"""
)
model_from_data_tables = calliope.Model(model_def)

# %% [markdown]
# ### Loading directly from in-memory dataframes
# If you create your tabular data in an automated manner in a Python script, you may want to load it directly into Calliope rather than saving it to file first.
# You can do that by setting `data` as the name of a key in a dictionary that you supply when you load the model:

# %%
model_def = read_rich_yaml(
    """
data_tables:
  tech_data:
    data: tech_data_df
    rows: parameters
    columns: techs
  tech_node_data:
    data: tech_node_data_df
    rows: [techs, nodes, parameters]
  tech_timestep_data:
    data: tech_timestep_data_df
    rows: timesteps
    columns: [techs, parameters]
  tech_carrier_data:
    data: tech_carrier_data_df
    rows: [techs, parameters]
    add_dims:
      carriers: electricity
  tech_cost_data:
    data: tech_cost_data_df
    rows: parameters
    columns: techs
    add_dims:
      costs: monetary
nodes:
  A.techs: {supply_tech, storage_tech, demand_tech}
  B.techs: {supply_tech, demand_tech}
"""
)
model_from_data_tables = calliope.Model(
    model_def,
    data_table_dfs={
        "tech_data_df": tech_data,
        # NOTE: inputs must be dataframes.
        # pandas Series objects must therefore be converted:
        "tech_node_data_df": tech_node_data.to_frame(),
        "tech_carrier_data_df": tech_carrier_data.to_frame(),
        "tech_timestep_data_df": tech_timestep_data,
        "tech_cost_data_df": tech_cost_data,
    },
)

# %% [markdown]
# ### Verifying model consistency
# We can solve both these simple models to check that their results are the same. First, we build and solve both models:

# %%
model_from_yaml.build(force=True)
model_from_yaml.solve(force=True)

# %%
model_from_data_tables.build(force=True)
model_from_data_tables.solve(force=True)

# %% [markdown]
# **Input data**. Now we check if the input data are exactly the same across both models:"

# %%
for variable_name, variable_data in model_from_yaml.inputs.data_vars.items():
    if variable_data.broadcast_equals(model_from_data_tables.inputs[variable_name]):
        print(f"Great work, {variable_name} matches")
    else:
        print(f"!!! Something's wrong! {variable_name} doesn't match !!!")


# %% [markdown]
# **Results**. And we check that the results also match exactly across both models:

# %%
for variable_name, variable_data in model_from_yaml.results.data_vars.items():
    if variable_data.broadcast_equals(model_from_data_tables.results[variable_name]):
        print(f"Great work, {variable_name} matches")
    else:
        print(f"!!! Something's wrong! {variable_name} doesn't match !!!")

# %% [markdown]
# ## Mixing YAML and data table definitions
# It is possible to only put some data into CSV files and define the rest in YAML.
# In fact, it almost always makes sense to build these hybrid definitions. For smaller models, you may only want to store timeseries data stored in CSV files and everything else in YAML:
#
# ```yaml
# data_tables:
#   tech_timestep_data:
#     data: outputs/loading_tabular_data/tech_timestep_data.csv
#     rows: timesteps
#     columns: [techs, parameters]
# techs:
#   supply_tech:
#     base_tech: supply
#     carrier_out: electricity
#     flow_cap_max: 10
#     cost_flow_cap:
#       data: 2
#       index: monetary
#       dims: costs
#
#   storage_tech:
#     base_tech: storage
#     carrier_in: electricity
#     carrier_out: electricity
#     flow_cap_max: 6
#     storage_cap_max: 7
#     cost_storage_cap:
#       data: 5
#       index: monetary
#       dims: costs
#     cost_flow_out:
#       data: 0.1
#       index: monetary
#       dims: costs
#
#   demand_tech:
#     base_tech: demand
#     carrier_in: electricity
#
#   transmission_tech:
#     base_tech: transmission
#     carrier_in: electricity
#     carrier_out: electricity
#     link_from: A
#     link_to: B
#     flow_cap_max: 8
#
# nodes:
#   A.techs: {supply_tech, storage_tech, demand_tech}
#   B.techs:
#     supply_tech:
#       flow_cap_max: 8
#     demand_tech:
# ```
#
# For larger models, with lots of nodes and / or technologies, it is increasingly easier to store other data such as technology and node definitions in the tabular CSV format too.
# This also helps to clean up things like the definition of technology costs, e.g.:
#
#
# ```yaml
# data_tables:
#   tech_timestep_data:
#     data: outputs/loading_tabular_data/tech_timestep_data.csv
#     rows: timesteps
#     columns: [techs, parameters]
#   tech_cost_data:
#     data: outputs/loading_tabular_data/tech_cost_data.csv
#     rows: parameters
#     columns: techs
#     add_dims:
#       costs: monetary
# techs:
#   supply_tech:
#     base_tech: supply
#     carrier_out: electricity
#     flow_cap_max: 10
#
#   storage_tech:
#     base_tech: storage
#     carrier_in: electricity
#     carrier_out: electricity
#     flow_cap_max: 6
#     storage_cap_max: 7
#
#   demand_tech:
#     base_tech: demand
#     carrier_in: electricity
#
#   transmission_tech:
#     base_tech: transmission
#     carrier_in: electricity
#     carrier_out: electricity
#     link_from: A
#     link_to: B
#     flow_cap_max: 8
#
# nodes:
#   A.techs: {supply_tech, storage_tech, demand_tech}
#   B.techs:
#     supply_tech:
#       flow_cap_max: 8
#     demand_tech:
# ```
#
# You can try these combinations - and others - yourself in this notebook and you will see that the result remains the same!

# %% [markdown]
# ## Overriding tabular data with YAML
#
# Another reason to mix tabular data with YAML is to allow you to keep track of overrides to specific parts of the model definition.
#
# For instance, we could change the number of a couple of parameters:
#
#
# ```yaml
# data_tables:
#   tech_data:
#     data: outputs/loading_tabular_data/tech_data.csv
#     rows: parameters
#     columns: techs
#   tech_node_data:
#     data: outputs/loading_tabular_data/tech_node_data.csv
#     rows: [techs, nodes, parameters]
#   tech_timestep_data:
#     data: outputs/loading_tabular_data/tech_timestep_data.csv
#     rows: timesteps
#     columns: [techs, parameters]
#   tech_carrier_data:
#     data: outputs/loading_tabular_data/tech_carrier_data.csv
#     rows: [techs, parameters]
#     add_dims:
#       carriers: electricity
#   tech_cost_data:
#     data: outputs/loading_tabular_data/tech_cost_data.csv
#     rows: parameters
#     columns: techs
#     add_dims:
#       costs: monetary
# techs:
#   storage_tech:
#     flow_cap_max: 5
#   transmission_tech:
#     flow_cap_max: 4
#
# nodes:
#   A.techs: {supply_tech, storage_tech, demand_tech}
#   B.techs: {supply_tech, demand_tech}
# ```
#

# %%
model_def = read_rich_yaml(
    """
data_tables:
  tech_data:
    data: outputs/loading_tabular_data/tech_data.csv
    rows: parameters
    columns: techs
  tech_node_data:
    data: outputs/loading_tabular_data/tech_node_data.csv
    rows: [techs, nodes, parameters]
  tech_timestep_data:
    data: outputs/loading_tabular_data/tech_timestep_data.csv
    rows: timesteps
    columns: [techs, parameters]
  tech_carrier_data:
    data: outputs/loading_tabular_data/tech_carrier_data.csv
    rows: [techs, parameters]
    add_dims:
      carriers: electricity
  tech_cost_data:
    data: outputs/loading_tabular_data/tech_cost_data.csv
    rows: parameters
    columns: techs
    add_dims:
      costs: monetary
techs:
  storage_tech:
    flow_cap_max: 5
  transmission_tech:
    flow_cap_max: 4

nodes:
  A.techs: {supply_tech, storage_tech, demand_tech}
  B.techs: {supply_tech, demand_tech}
"""
)
model_from_data_tables_w_override = calliope.Model(model_def)

# Let's compare the two after overriding `flow_cap_max`
flow_cap_old = model_from_data_tables.inputs.flow_cap_max.to_series().dropna()
flow_cap_new = (
    model_from_data_tables_w_override.inputs.flow_cap_max.to_series().dropna()
)
pd.concat([flow_cap_old, flow_cap_new], axis=1, keys=["old", "new"])

# %% [markdown]
# We can also switch off technologies / nodes that would otherwise be introduced by our data tables:
#
#
# ```yaml
# data_tables:
#   tech_data:
#     data: outputs/loading_tabular_data/tech_data.csv
#     rows: parameters
#     columns: techs
#   tech_node_data:
#     data: outputs/loading_tabular_data/tech_node_data.csv
#     rows: [techs, nodes, parameters]
#   tech_timestep_data:
#     data: outputs/loading_tabular_data/tech_timestep_data.csv
#     rows: timesteps
#     columns: [techs, parameters]
#   tech_carrier_data:
#     data: outputs/loading_tabular_data/tech_carrier_data.csv
#     rows: [techs, parameters]
#     add_dims:
#       carriers: electricity
#   tech_cost_data:
#     data: outputs/loading_tabular_data/tech_cost_data.csv
#     rows: parameters
#     columns: techs
#     add_dims:
#       costs: monetary
# techs:
#   storage_tech:
#     active: false
#
# nodes:
#   A.techs: {supply_tech, storage_tech, demand_tech}
#   B.techs:
#     supply_tech:
#     demand_tech:
#       active: false
# ```

# %%
model_def = read_rich_yaml(
    """
data_tables:
  tech_data:
    data: outputs/loading_tabular_data/tech_data.csv
    rows: parameters
    columns: techs
  tech_node_data:
    data: outputs/loading_tabular_data/tech_node_data.csv
    rows: [techs, nodes, parameters]
  tech_timestep_data:
    data: outputs/loading_tabular_data/tech_timestep_data.csv
    rows: timesteps
    columns: [techs, parameters]
  tech_carrier_data:
    data: outputs/loading_tabular_data/tech_carrier_data.csv
    rows: [techs, parameters]
    add_dims:
      carriers: electricity
  tech_cost_data:
    data: outputs/loading_tabular_data/tech_cost_data.csv
    rows: parameters
    columns: techs
    add_dims:
      costs: monetary
techs:
  storage_tech:
    active: false
nodes:
  A.techs: {supply_tech, storage_tech, demand_tech}
  B.techs:
    supply_tech:
    demand_tech:
      active: false
"""
)
model_from_data_tables_w_deactivations = calliope.Model(model_def)

# Let's compare the two after overriding `flow_cap_max`
definition_matrix_old = (
    model_from_data_tables.inputs.definition_matrix.to_series().dropna()
)
definition_matrix_new = (
    model_from_data_tables_w_deactivations.inputs.definition_matrix.to_series().dropna()
)
pd.concat([definition_matrix_old, definition_matrix_new], axis=1, keys=["old", "new"])
