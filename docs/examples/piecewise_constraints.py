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
#     display_name: calliope_docs_build [conda env:calliope-docs-new]
#     language: python
#     name: conda-env-calliope-docs-new-calliope_docs_build
# ---

# %% [markdown]
# # Defining piecewise linear constraints
#
# In this tutorial, we use the national scale example model to implement a piecewise linear constraint.
# This constraint will represent a non-linear relationship between capacity and cost per unit capacity of Concentrating Solar Power (CSP).

# %%

import numpy as np
import plotly.express as px

import calliope
from calliope.io import read_rich_yaml

calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# # Model setup

# %% [markdown]
# ## Defining our piecewise curve
#
# In the base national scale model, the CSP has a maximum rated capacity of 10,000 kW and a cost to invest in that capacity of 1000 USD / kW.
#
# In our updated model, the cost to invest in capacity will vary from 5000 USD / kW to 500 USD / kW as the CSP capacity increases:

# %%
capacity_steps = [0, 2500, 5000, 7500, 10000]
cost_steps = [0, 3.75e6, 6e6, 7.5e6, 8e6]

cost_per_cap = np.nan_to_num(np.divide(cost_steps, capacity_steps)).astype(int)

fig = px.line(
    x=capacity_steps,
    y=cost_steps,
    labels={"x": "Capacity (kW)", "y": "Investment cost (USD)"},
    markers="o",
    range_y=[0, 10e6],
    text=[f"{i} USD/kW" for i in cost_per_cap],
)
fig.update_traces(textposition="top center")
fig.show()


# %% [markdown]
# We can then provide this data when we load our model:
#
# <div class="admonition note">
#     <p class="admonition-title">Note</p>
#     <p>
#         We must index our piecewise data over "breakpoints".
#     </p>
# </div>
#

# %%
new_params = f"""
  parameters:
    capacity_steps:
      data: {capacity_steps}
      index: [0, 1, 2, 3, 4]
      dims: "breakpoints"
    cost_steps:
      data: {cost_steps}
      index: [0, 1, 2, 3, 4]
      dims: "breakpoints"
"""
print(new_params)
new_params_as_dict = read_rich_yaml(new_params)
m = calliope.examples.national_scale(override_dict=new_params_as_dict)

# %%
m.inputs.capacity_steps

# %%
m.inputs.cost_steps

# %% [markdown]
# ## Creating our piecewise constraint
#
# We create the piecewise constraint by linking decision variables to the piecewise curve we have created.
# In this example, we need:
# 1. a new decision variable for investment costs that can take on the value defined by the curve at a given value of `flow_cap`;
# 1. to link that decision variable to our total cost calculation; and
# 1. to define the piecewise constraint.

# %%
new_math = """
  variables:
    piecewise_cost_investment:
      description: "Investment cost that increases monotonically"
      foreach: ["nodes", "techs", "carriers", "costs"]
      where: "[csp] in techs"
      bounds:
        min: 0
        max: .inf
      default: 0
  global_expressions:
    cost_investment_flow_cap:
      equations:
        - expression: "$cost_sum * flow_cap"
          where: "NOT [csp] in techs"
        - expression: "piecewise_cost_investment"
          where: "[csp] in techs"
  piecewise_constraints:
    csp_piecewise_costs:
      description: "Set investment costs values along a piecewise curve using special ordered sets of type 2 (SOS2)."
      foreach: ["nodes", "techs", "carriers", "costs"]
      where: "piecewise_cost_investment"
      x_expression: "flow_cap"
      x_values: "capacity_steps"
      y_expression: "piecewise_cost_investment"
      y_values: "cost_steps"
"""

# %% [markdown]
# # Building and checking the optimisation problem
#
# With our piecewise constraint defined, we can build our optimisation problem and inject this new math.

# %%
new_math_as_dict = read_rich_yaml(new_math)
m.build(add_math_dict=new_math_as_dict)

# %% [markdown]
# And we can see that our piecewise constraint exists in the built optimisation problem "backend"

# %%
m.backend.verbose_strings()
m.backend.get_piecewise_constraint("csp_piecewise_costs").to_series().dropna()

# %% [markdown]
# ## Solve the optimisation problem
#
# Once we have all of our optimisation problem components set up as we desire, we can solve the problem.

# %%
m.solve()

# %% [markdown]
# The results are stored in `m._model_data` and can be accessed by the public property `m.results`

# %% [markdown]
# ## Analysing the outputs

# %%
# Absolute
csp_cost = m.results.cost_investment_flow_cap.sel(techs="csp")
csp_cost.to_series().dropna()

# %%
# Relative to capacity
csp_cap = m.results.flow_cap.sel(techs="csp")
csp_cost_rel = csp_cost / csp_cap
csp_cost_rel.to_series().dropna()

# %%
# Plotted on our piecewise curve
fig.add_scatter(
    x=csp_cap.to_series().dropna().values,
    y=csp_cost.to_series().dropna().values,
    mode="markers",
    marker_symbol="cross",
    marker_size=10,
    marker_color="cyan",
    name="Installed capacity",
)
fig.show()

# %% [markdown]
# ## Troubleshooting
#
# If you are failing to load a piecewise constraint or it isn't working as expected, here are some common things to note:
#
# 1. The extent of your `x_values` and `y_values` will dictate the maximum values of your piecewise decision variables.
# In this example, we define `capacity_steps` over the full capacity range that we allow our CSP to cover in the model.
# However, if we set `capacity_steps` to `[0, 2500, 5000, 7500, 9000]` then `flow_cap` would _never_ go above a value of 9000.
#
# 2. The `x_values` and `y_values` parameters must have the same number of breakpoints and be indexed over `breakpoints`.
# It is possible to extend these parameters to be indexed over other dimensions (e.g., different technologies with different piecewise curves) but it must _always_ include the `breakpoints` dimension.
#
# 3. `x_values` must increase monotonically. That is, `[0, 5000, 2500, 7500, 10000]` is not valid for `capacity_steps` in this example.
# `y_values`, on the other hand, _can_ vary any way you like; `[0, 6e6, 3.75e6, 8e6, 7.5e6]` is valid for `cost_steps`.
#
# 4. `x_expression` and `y_expression` _must_ include reference to at least one decision variable.
# It can be a math expression, not only a single decision variable. `flow_cap + storage_cap / 2` would be valid for `x_expression` in this example.
#
# 5. Piecewise constraints will make your problem more difficult to solve since each breakpoint adds a binary decision variable.
# Larger models with detailed piecewise constraints may not solve in a reasonable amount of time.
#
