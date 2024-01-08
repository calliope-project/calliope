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
#     display_name: calliope_dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mixed Integer Linear Programming (MILP) example model

# %%
import calliope
from calliope.util.tools import yaml_snippet

# We increase logging verbosity
calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# This example is based on the Urban scale example model, but with an override to introduce binary and integer variables.
# This override is applied from the `scenarios.yaml` file:

# %%
yaml_snippet("example_models/urban_scale/scenarios.yaml", "milp")

# %% [markdown]
# <div class="admonition note">
#     <p class="admonition-title">Note</p>
#     <p>
#         MILP functionality can be easily applied, but convergence is slower as a result of integer/binary variables.
#         It is recommended to use a commercial solver (e.g. Gurobi, CPLEX) if you wish to utilise these variables outside this example model.
#     </p>
# </div>

# %% [markdown]
# ## Model definition
#
# We will only discuss the components of the model definition that differ from the urban scale example model.
# Refer to that tutorial page for more information on this model.

# %% [markdown]
# ### Units
#
# The capacity of a technology is usually a continuous decision variable, which can be within the range of 0 and `flow_cap_max` (the maximum capacity of a technology).
# In this model, we introduce a unit limit on the CHP instead:

# %%
yaml_snippet("example_models/urban_scale/scenarios.yaml", "chp")

# %% [markdown]
# A unit maximum allows a discrete, integer number of CHP to be purchased, each having a capacity of `flow_cap_per_unit`.
# `flow_cap_max` and `flow_cap_min` are now ignored, in favour of `units_max` or `units_min`.
#
# A useful feature unlocked by introducing this is the ability to set a minimum operating capacity which is *only* enforced when the technology is operating.
# In the LP model, `flow_out_min_relative` would force the technology to operate at least at that proportion of its maximum capacity at each time step.
# In this model, the newly introduced `flow_out_min_relative` of 0.2 will ensure that the output of the CHP is 20% of its maximum capacity in any time step in which it has a _non-zero output_.

# %% [markdown]
# ### Purchase cost
#
# The boiler does not have a unit limit, it still utilises the continuous variable for its capacity. However, we have introduced a `purchase` cost:

# %%
yaml_snippet("example_models/urban_scale/scenarios.yaml", "boiler")

# %% [markdown]
# By introducing this, the boiler is now associated with a binary decision variable.
# It is 1 if the boiler has a non-zero `flow_cap` (i.e. the optimisation results in investment in a boiler) and 0 if the capacity is 0.
#
# The purchase cost is applied to the binary result, providing a fixed cost on purchase of the technology, irrespective of the technology size.
# In physical terms, this may be associated with the cost of pipework, land purchase, etc.
# The purchase cost is also imposed on the CHP, which is applied to the number of integer CHP units in which the solver chooses to invest.

# %% [markdown]
# ### Asynchronous flow in/out
#
# The heat pipes which distribute thermal energy in the network may be prone to dissipating heat in an "unphysical" way.
# I.e. given that they have distribution losses associated with them, in any given timestep a link could produce and consume energy simultaneously.
# It would therefore lose energy to the atmosphere, but have a net energy transmission of zero.
#
# This allows e.g. a CHP facility to overproduce heat to produce more cheap electricity, and have some way of dumping that heat.
# The `async_flow_switch` binary variable (triggered by the `force_async_flow` parameter) ensures this phenomenon is avoided:

# %%
yaml_snippet("example_models/urban_scale/scenarios.yaml", "heat_pipes")

# %% [markdown]
# Now, only one of `flow_out` and `flow_in` can be non-zero in a given timestep.
# This constraint can also be applied to storage technologies, to similarly control charge/discharge.

# %% [markdown]
# ## Load model and examine inputs

# %%
model = calliope.examples.milp()

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
# Integer/binary variables `units`, `purchased`, etc. are available in the results

# %%
model.results.units.to_series().dropna()

# %%
model.results.purchased.to_series().dropna()

# %%
model.results.operating_units.to_series().dropna().unstack("techs").head()

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
