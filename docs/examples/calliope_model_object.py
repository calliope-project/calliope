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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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
# ## Model definition dictionary
#
# `m._model_def_dict` is a python dictionary that holds all the data from the model definition YAML files, restructured into one dictionary.
#
# The underscore before the method indicates that it defaults to being hidden (i.e. you wouldn't see it by trying a tab auto-complete and it isn't documented)

# %%
m._model_def_dict.keys()

# %% [markdown]
# `techs` hold only the information about a technology that is specific to that node

# %%
m._model_def_dict["techs"]["pv"]

# %% [markdown]
# `nodes` hold only the information about a technology that is specific to that node

# %%
m._model_def_dict["nodes"]["X2"]["techs"]["pv"]

# %% [markdown]
# ## Model data
#
# `m._model_data` is an xarray Dataset.
# Like `_model_def_dict` it is a hidden prperty of the Model as you are expected to access the data via the public property `inputs`

# %%
m.inputs

# %% [markdown]
# Until we solve the model, `inputs` is the same as `_model_data`

# %%
m._model_data

# %% [markdown]
# We can find the same PV `flow_cap_max` data as seen in `m._model_run`

# %%
m._model_data.flow_cap_max.sel(techs="pv").to_series().dropna()

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
m.backend.update_parameter("flow_cap_max", m.inputs.flow_cap_max * 2)
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
# The results are stored in `m._model_data` and can be accessed by the public property `m.results`

# %%
m.results

# %% [markdown]
# We can also view the data within the backend directly

# %%
m.backend.get_variable("flow_cap", as_backend_objs=False).to_series().dropna()

# %% [markdown]
# # Save

# %%
# We can save at any point, which will dump the entire m._model_data to file.
# NetCDF is recommended, as it retains most of the data and can be reloaded into a Calliope model at a later date.


output_path = Path(".") / "outputs" / "4_calliope_model_object"
output_path.mkdir(parents=True, exist_ok=True)

m.to_netcdf(output_path / "example.nc")  # Saves a single file
m.to_csv(
    output_path / "csv_files", allow_overwrite=True
)  # Saves a file for each xarray DataArray
