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
# # The Calliope model and backend objects
#
# In this tutorial, we use the urban scale example model to go into a bit more detail on the public and non-public properties of the `calliope.Model` and `calliope.Model.backend` objects.

# %%
from pathlib import Path

import calliope

calliope.set_log_verbosity("INFO", include_solver_output=False)

# %% [markdown]
# # Model input

# %%
# Initialise the model with the Urban Scale example model
m = calliope.examples.urban_scale()

# %%
# Get information on the model
print(m.info())

# %% [markdown]
# ## Model data
#
# `m.inputs` and `m.results` are xarray Datasets.

# %%
m.inputs

# %% [markdown]
# Until we solve the model, `results` is empty.

# %%
m.results

# %% [markdown]
# # Building and checking the optimisation problem
#
# Calling `m.build` allows us to build the optimisation problem, which creates arrays of Python objects from the YAML math formulation.

# %%
m.build()

# %% [markdown]
# As with the calliope `Model`, the backend has its own dataset containing all the arrays of backend objects

# %%
m.backend._dataset

# %% [markdown]
# There is then a public API to access filtered views on this dataset, e.g. input parameters...

# %%
m.backend.parameters

# %% [markdown]
# ... or constraints

# %%
m.backend.constraints

# %% [markdown]
# You can also access backend arrays in text format, to debug the problem:

# %%
m.backend.get_constraint(
    "area_use_capacity_per_loc", as_backend_objs=False
).to_pandas().dropna(how="all", axis=0)

# %% [markdown]
# We can increase the verbosity of the constraint/global expression "body" by calling the backend method `verbose_strings`.
# We do not do this automatically as it entails a memory/time overhead on building the model and is only necessary for debugging your optimisation problem.

# %%
m.backend.verbose_strings()
m.backend.get_constraint(
    "area_use_capacity_per_loc", as_backend_objs=False
).to_pandas().dropna(how="all", axis=0)

# %% [markdown]
# ## Updating the optimisation problem in-place
#
# If we want to update a parameter value or fix a decision variable, we can do so now that we have built the optimisation problem

# %%
m.backend.update_input("flow_cap_max", m.inputs.flow_cap_max * 2)
m.backend.get_parameter("flow_cap_max", as_backend_objs=False).sel(
    techs="pv"
).to_series().dropna()

# %% [markdown]
# ## Solve the optimisation problem
#
# Once we have all of our optimisation problem components set up as we desire, we can solve the problem.

# %%
m.solve()

# %% [markdown]
# The results can now be accessed by the public property `m.results`

# %%
m.results

# %% [markdown]
# We can also view the data within the backend directly

# %%
m.backend.get_variable("flow_cap", as_backend_objs=False).to_series().dropna()

# %% [markdown]
# # Save

# %%
# We can save at any point, which will dump the entire model to file.
# NetCDF is recommended, as it retains the model data _and_ attributes and can be reloaded into a Calliope model at a later date.


output_path = Path(".") / "outputs" / "4_calliope_model_object"
output_path.mkdir(parents=True, exist_ok=True)

m.to_netcdf(
    output_path / "example.nc"
)  # Saves a single file with two groups: `inputs` and `results`
m.to_csv(
    output_path / "csv_files", allow_overwrite=True
)  # Saves a file for each xarray DataArray
