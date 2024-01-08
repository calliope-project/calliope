# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: calliope_dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running the urban scale example model
# This notebook will show you how to load, build, solve, and examine the results of the urban scale example model.

# %%
import calliope

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
# You can apply node/tech/carrier only operations, like summing information over nodes

# %%
df_heat.info()

# %% [markdown]
# We can also examine total technology costs.

# %%
costs = model.results.cost.to_series().dropna()
costs

# %% [markdown]
# We can also examine levelized costs for each location and technology, which is calculated in a post-processing step.

# %%
lcoes = (
    model.results.systemwide_levelised_cost.sel(carriers="electricity")
    .to_series()
    .dropna()
)
lcoes

# %% [markdown]
# ---

# %% [markdown]
# See the [Calliope documentation](https://calliope.readthedocs.io/) for more details on setting up and running a Calliope model.
