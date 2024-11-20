# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Generating SPORES
# An interactive example of how to generate near-optimal system designs (or SPORES) out of a Calliope v0.7.0 model. This example relies solely on default software functionality and a custom Python function to determine how to assign penalties (scores) to previously explored system design options.

# %%
# Importing the required packages
import calliope
import xarray as xr

# %% [markdown]
# ## Cost-optimal model run and extraction of SPORES-relevant outputs

# %%
# Loading model files and building the model
model = calliope.Model(
    "example_models/latest_national_scale/model.yaml", scenario="simplified_spores"
)
model.build()

# Solving
model.solve()

# Extracting SPORES-relevant data
least_feasible_cost = model.results.cost.loc[{"costs": "monetary"}].sum().sum()
print("The minimum cost for a feasible system design is {}".format(least_feasible_cost.values))


# %% [markdown]
# ## SPORES model run
# ### Definition of the penalty-assignment methods

# %%
def scoring_integer(results, backend):
    # Filter for technologies of interest
    spores_techs = backend.inputs["spores_tracker"].notnull()
    # Look at capacity deployment in the previous iteration
    previous_cap = results.flow_cap 
    # Make sure that penalties are applied only to non-negligible deployments of capacity
    min_relevant_size = 0.1 * previous_cap.where(spores_techs).max(
        ["nodes", "carriers", "techs"]
    )
    # Where capacity was deployed more than the minimal relevant size, assign an integer penalty (score)
    new_score = previous_cap.copy()
    new_score = new_score.where(spores_techs, other=0)
    new_score = new_score.where(new_score > min_relevant_size, other=0)
    new_score = new_score.where(new_score == 0, other=1000)
    # Transform the score into a "cost" parameter
    new_score.rename("cost_flow_cap")
    new_score = new_score.expand_dims(costs=["spores_score"]).copy()
    new_score = new_score.sum("carriers")
    # Extract the existing cost parameters from the backend
    all_costs = backend.get_parameter("cost_flow_cap", as_backend_objs=False)
    try:
        all_costs = all_costs.expand_dims(nodes=results.nodes).copy()
    except:
        pass
    # Create a new version of the cost parameters by adding up the calculated scores
    new_all_costs = all_costs
    new_all_costs.loc[{"costs":"spores_score"}] += new_score.loc[{"costs":"spores_score"}]

    return new_all_costs


# %% [markdown]
# ### Iterating over the desired number of alternatives

# %%
# Create some lists to store results as they get generated
spores = [] # full results
scores = [] # scores only
spores_counter = 1
number_of_spores = 5

# %%
for i in range(spores_counter, spores_counter + number_of_spores):

    if spores_counter == 1:
        # Store the cost-optimal results
        spores.append(model.results.expand_dims(spores=[0]))
        scores.append(
            model.backend.get_parameter("cost_flow_cap", as_backend_objs=False)
            .sel(costs="spores_score")
            .expand_dims(spores=[0])
        )
        # Update the slack-cost backend parameter based on the calculated minimum feasible system design cost
        model.backend.update_parameter("spores_cost_max", least_feasible_cost)
        # Update the objective_cost_weights to reflect the ones defined for the SPORES mode
        model.backend.update_parameter(
            "objective_cost_weights", model.inputs.spores_objective_cost_weights
        )
    else:
        pass

    # Calculate weights based on a scoring method
    spores_score = scoring_integer(model.results, model.backend)
    # Assign a new score based on the calculated penalties
    model.backend.update_parameter(
        "cost_flow_cap", spores_score.reindex_like(model.inputs.cost_flow_cap)
    )
    # Run the model again to get a solution that reflects the new penalties
    model.solve(force=True)
    # Store the results
    spores.append(model.results.expand_dims(spores=[i]))
    scores.append(
        model.backend.get_parameter("cost_flow_cap", as_backend_objs=False)
        .sel(costs="spores_score")
        .expand_dims(spores=[i])
    )

    spores_counter += 1
        
# Concatenate the results in the storage lists into xarray objects 
spore_ds = xr.concat(spores, dim="spores")
score_da = xr.concat(scores, dim="spores")

# %% [markdown]
# ## Plotting and sense-check

# %%
# Import plotting libraries
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
# Extract the deployed capacities across SPORES, which we want to visualise
flow_caps = spore_ds.flow_cap.where(
    model.backend.inputs["spores_tracker"].notnull()).sel(
        carriers='power').to_series().dropna().unstack("spores")
flow_caps

# %%
# Plot the capacities per location across the generated SPORES
ax = plt.subplot(111)
colors = mpl.colormaps['Pastel1'].colors
flow_caps.plot.bar(
    ax=ax, ylabel="Capacity (kW)", ylim=[0, 40000],
    color=colors[0:len(flow_caps.columns)]
)

# %%
