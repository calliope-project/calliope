## 0.7.0.dev5 (Unreleased)

## 0.7.0.dev4 (2024-09-10)

### User-facing changes

|fixed| Decision variable domain in math docs to use $\in$ instead of $\forall$ (#652).

|fixed| Clarity of `flow_cap_min` description in documentation (#653).

|changed| API/schema documentation is de-ranked in documentation search bar results (#670).

|new| Math component cross-references in both directions ("uses" and "used in") in Markdown math documentation (#643).

|fixed| Duplicated links in math documentation (#651).

|changed| `node_groups` and `tech_groups` changed to a general top-level `templates` key,
accessed via the `template` key (replacing `inherit`) in `nodes` and `techs` (#600).

|fixed| Contribution of `cost_om_annual_investment_fraction` to total investment costs, to not apply to depreciated costs (#645).

|fixed| Math for multi-carrier variable export costs (#663).

|new| Piecewise constraints added to the YAML math with its own unique syntax (#107).
These constraints will be added to the optimisation problem using Special Ordered Sets of Type 2 (SOS2) variables.

|new| Direct interface to the Gurobi Python API using `!#yaml config.build.backend: gurobi` or `!#python model.build(backend="gurobi")`.
Tests show that using the gurobi solver via the Python API reduces peak memory consumption and runtime by at least 30% for the combined model build and solve steps.
This requires the `gurobipy` package which can be installed with `mamba`: `mamba install gurobi::gurobi`.

|fixed| Force a header row in tabular data loaded from CSV to (#596).
Fixes issue where unexpected index levels can end up in the loaded data (#573).

|fixed| Single element lists/sets in the model Dataset attribute dictionary are restored to lists/sets on loading from NetCDF (#614).

|new| Decision variables and global expressions can have a `title` defined, which will be available in the model results as attributes of those components and can be used for e.g. visualisation (#582).
Parameter titles from the model definition schema will also propagate to the model inputs.

|fixed| Backend parameter updates propagate correctly through global expressions in the order those expressions were defined (#616).

|fixed| If setting `model.backend.verbose_strings()`, rebuilt model components from making backend parameter updates will automatically have verbose strings (#623).

|fixed| Erroneous use of `dimensions:` in docs example of an indexed parameter (#612).

|changed| `add_dimensions` to `add_dims` in `data_sources` definition to align with `dims` in indexed parameter definition (#621).

|new| Allow extracting shadow prices into results by listing constraints in `config.solve.shadow_prices`, e.g. `config.solve.shadow_prices: ["system_balance"]`  Shadow prices will be added as variables to the model results as `shadow_price_{constraintname}`, e.g. `shadow_price_system_balance`.

|new| Model stores key timestamps as attributes:

* `timestamp_model_creation`: at the start of `Model.__init__()`
* `timestamp_build_started`: at the start of `Model.build()`
* `timestamp_build_complete`: at the end of `Model.build()`
* `timestamp_solve_started`: at the start of `Model.solve()`
* `timestamp_solve_complete`: at the end of `Model.solve()`

### Internal changes

|new| `gurobipy` is a development dependency that will be added as an optional dependency to the conda-forge calliope feedstock recipe.

|changed| Added any new math dicts defined with `calliope.Model.backend.add_[...](...)` to the backend math dict registry stored in `calliope.Model.backend.inputs.attrs["math"]`.

|changed| Function vectorisation when creating backend component arrays uses `numpy.frompyfunc` instead of `xarray.apply_ufunc`, so that masking with a `where` array can be done at function calltime.
This requires pre-broadcasting all arrays being passed to the vectorised function, but reduces memory peaks in the Gurobi backend interface in particular.

|changed| Default parameter values are not used to fill NaNs when adding parameters to the backend, but when evaluating expressions.
This reduces memory footprint of parameter arrays.

|fixed| Removed unused debug parameter in `Model.__init__()`

|changed| Ruff linter checking was extended with pydocstrings, flake8-pytest, and pyupgrade.

|changed| Moved from black formatting to the Ruff formatter (black-based, but faster).

## 0.7.0.dev3 (2024-02-14)

### User-facing changes

|new| `mkdocs_tabbed` option when writing math documentation to file (`calliope.Model.math_documentation.write(...)`) which will add YAML snippets to all rendered math as a separate "tab" if writing to Markdown.
Requires the [PyMdown tabbed extension](https://facelessuser.github.io/pymdown-extensions/extensions/tabbed/) to render the tabs correctly in an [MkDocs](https://www.mkdocs.org/) project.

|new| List of pre-defined parameters given in the `pre-defined` math documentation, with references back to the constraints/variables/global expressions in which they are defined (either in the `expression` string or the `where` string).

|new| Units and default values for variables and global expressions added to the math documentation.

|new| Variables and global expressions can have a `default` value, which is used to fill missing array elements when doing math operations.
These default values ensure that `NaN` doesn't creep into the built optimisation problem math and are set to values that lead to them having no impact on the optimal solution.

|new| Utility function `calliope.util.schema.update_model_schema(...)` to add user-defined parameters to the model schema / update existing parameters using YAML schema syntax.
`calliope.util.schema.reset()` can be used to clean the model schema and return to the original, pre-defined schema.

|fixed| Timeseries clustering file can be a non-ISO standard date format.
Both the index and the values of the timeseries (both being date strings) should be in the user-defined `config.init.time_format`.

|fixed| the decision variable `purchased_units` is linked to `flow_cap` even if neither of the parameters `flow_cap_min` or `flow_cap_max` have been defined by the user.

|changed| `inbuilt` math -> `pre-defined` math and `custom` math -> `pre-defined` math in the documentation.

|changed| Calliope attribute dictionaries (AttrDicts) no longer sort dictionary keys on `union`. Key order is now: original dictionary key order + any new keys being added in the order they appear in the new dictionary.

|fixed| Dimensions with numeric data can be defined in tabular data _or_ YAML and will appear as numeric in the processed Calliope model input dataset.
If all dimension data can be coerced to a numeric data type (e.g. `["10", 100, "-1"]`), then it _will_ be coerced (e.g., `[10, 100, -1]`).

### Internal changes

|new| `py.typed` file so that mypy recognises Calliope as a typed library when it is imported as a dependency.

|fixed| Spelling of Black config option `skip-magic-trailing-comma`.

## 0.7.0.dev2 (2024-01-26)

v0.7 includes a major change to how Calliope internally operates.
Along with this, there are multiple changes to how Calliope models are defined and configured.
This requires adapting models to work with 0.7.
We group changes into those that are primarily user-facing and relevant for all Calliope users, and those that are primarily internal, and relevant only for Calliope developers.

### User-facing changes

This section gives a brief summary of changes.
For more detail, see our migrating from v0.6 to v0.7 section in our [documentation](https://calliope.readthedocs.io/en/latest/migrating/).

|new| Storage buffers available in all technology base classes.

|new| Multiple carriers and different carriers in/out available in all technology base classes.

|new| `node_groups` added to match `tech_groups`.

|new| technology efficiencies split into inflow and outflow efficiencies.

|new| Technology capacities and efficiencies can be differentiated between technology carriers.

|new| Parameters can be defined outside the scope of `nodes` and `techs` using the top-level `parameters` key in YAML.

|new| Any parameters can be indexed over arbitrary dimensions, both core Calliope dimensions and new, user-defined dimensions.

|new| Non-timeseries data can be loaded from CSV files or in-memory Pandas DataFrames using the top-level `data_sources` key in YAML.

|new| User-defined mathematical formulations using the new Calliope math syntax can be loaded when creating a model.

|new| Shadow prices obtained from a dual LP problem can be read by using `model.backend.shadow_prices.get("constraint_name")`.

|changed| |backwards-incompatible| Updated to support Python versions >= 3.10.

|changed| |backwards-incompatible| Updated to Pandas >= v2.1, Pyomo >= v6.4, Xarray >= v2023.10.

|changed| |backwards-incompatible| Flat technology definitions, removing the distinction between `essentials`, `constraints` and `costs`.

|changed| |backwards-incompatible| Timeseries data is defined under the `data_sources` top-level key, not with `file=`/`df=` at the technology level.

|changed| |backwards-incompatible| Demand and carrier consumption values are strictly positive instead of strictly negative.

|changed| |backwards-incompatible| `model.run()` method → two-stage `model.build()` + `model.solve()` methods.

|changed| |backwards-incompatible| `model` and `run` top-level keys → `config.init`/`.build`/`.solve`.

|changed| |backwards-incompatible| `locations` top-level key and `loc` data dimensions → `nodes`.

|changed| |backwards-incompatible| `parent` technology parameter → `base_tech` + `inherit`.

|changed| |backwards-incompatible| Cost parameters are flattened and use the indexed parameter syntax.

|changed| |backwards-incompatible| `links` top-level key → transmission links defined in `techs`.

|changed| |backwards-incompatible| Various parameters/decision variables renamed (namely `energy_` → `flow_`, `carrier_prod`/`_con` → `flow_out`/`_in`, and `resource` → `source_use`/`sink_use`).

|changed| |backwards-incompatible| Various configuration options renamed.

|changed| |backwards-incompatible| `force_resource` technology parameter → `source_use_equals` / `sink_use_equals`.

|changed| |backwards-incompatible| `units` + `purchased` decision variables → `purchased_units`.

|changed| |backwards-incompatible| Parameters added to explicitly trigger MILP and storage decision variables/constraints.

|changed| |backwards-incompatible| Structure of input and output data within a Calliope model updated to remove concatenated `loc::tech::carrier` sets and technology subsets (e.g. `techs_supply`) in favour of sparse arrays indexed over core dimensions only (`nodes`, `techs`, `carriers`, `timesteps`).

|changed| |backwards-incompatible| `coordinates.lat`/`lon` node parameter → `latitude`/`longitude`.

|changed| |backwards-incompatible| Distance units default to kilometres and can be reverted to metres with `config.init.distance_unit`.

|changed| |backwards-incompatible| `operate` mode input parameters now expected to match `plan` mode decision variable names (e.g., `flow_cap`).

|changed| |backwards-incompatible| Cyclic storage is defined per-technology, not as a top-level configuration option.

|changed| Documentation has been overhauled.

|removed| `_equals` constraints.
Use both `_min` and `_max` constraints to the same value.

|removed| `x`/`y` coordinates.
Use geographic lat/lon coordinates (in `EPSG:4326` projection) instead.

|removed| Comma-separated node definitions.
Inheriting duplicate definitions from `node_groups` instead.

|removed| `supply_plus` and `conversion_plus` technology base classes.
Use `supply` and `conversion` technology base classes instead.

|removed| `carrier` key.
Use `carrier_in` and `carrier_out` instead.

|removed| `carrier_tiers` and `carrier_ratios`.
Use indexed parameter definitions for `flow_out_eff` and your own math instead.

|removed| `calliope.Model.get_formatted_array`.
The Calliope internal representation of data now matches the format achieved by calling this method in v0.6.

|removed| Group constraints.
Use your own math instead.

|removed| Configuration options for features we no longer support.

|removed| Plotting.
See our documentation for example of how to visualise your data with Plotly.

|removed| Clustering.
Cluster your data before creating your Calliope model.
Mapping of timeseries dates to representative days is still possible.

### Internal changes

|new| Automatic release uploads to PyPI and new accompanying pre-release pipeline.

|new| Choice of issue templates.

|new| YAML schema to catch the worst offences perpetrated in the model definition / configuration.
This schema is also rendered as a reference page in the documentation, replacing `defaults`/`config` tables.

|new| The model mathematical formulation (constraints, decision variables, objectives) is stored in a YAML configuration file: `math/base.yaml`.
Equation expressions and the logic to decide on when to apply a constraint/create a variable etc. are given in string format.
These strings are parsed according to a set of documented rules.

|changed| Development environment installation instructions (they're now simpler!).

|changed| Documentation has been ported to Markdown pages and is built using MKDocs and the Material theme.

|changed| Pre-processed model data checks are conducted according to a YAML configuration, instead of a hard-coded set of python functions.
An API will be created in due course to allow the user to add their own checks to the configuration.

|changed| Costs are now Pyomo expressions rather than decision variables.

|changed| When a model is loaded into an active session, configuration dictionaries are stored as dictionaries instead of serialised YAML strings in the model data attributes dictionary.
Serialisation and de-serialisation only occur on saving and loading from NetCDF, respectively.

|changed| Backend interface has been abstracted to enable non-Pyomo solver interfaces to be implemented in due course.

|changed| Repository structure has been re-configured to use the `src` layout, to rely on the `pyproject.toml` configuration file for most config, and to use only `.txt` requirements files (for pip+conda cross-compatibility)

|changed| CI moved from Azure pipelines (back) to GitHub Actions.

|changed| Stronger reliance on `pre-commit`, including a CI check to run it in Pull Requests.

## 0.6.10 (2023-01-18)

|changed| |backwards-incompatible| Updated to Numpy v1.23, Pandas v1.5, Pyomo v6.4, Ruamel.yaml v0.17, Scikit-learn v1.2, Xarray v2022.3, GLPK v5. This enables Calliope to be installed on Apple Silicon devices, but changes the result of algorithmic timeseries clustering. [In scikit-learn version 0.24.0, the method of random sampling for K-Means clustering was changed](https://scikit-learn.org/0.24/whats_new/v0.24.html#changed-models). This change will lead to different optimisation results if using [K-Means clustering](https://calliope.readthedocs.io/en/v0.6.10/user/advanced_features.html#time-resolution-adjustment) in your model.

|changed| |backwards-incompatible| Removed support for Python version 3.7 since some updated dependencies are not available in this version.

|changed| Installation instructions for developers have changed since we no longer duplicate pinned packages between the development/testing requirements file (`requirements.yml`) and the package requirements file (`requirements.txt`). See [the documentation](https://calliope.readthedocs.io/en/v0.6.10/user/installation.html) for updated instructions.

|fixed| Set ordering in the model dataset is consistent before and after optimising a model with clustered timeseries. Previously, the link between clusters and timesteps would become mixed following optimisation, so running `model.run(force_rerun=True)` would yield a different result.

## 0.6.9 (2023-01-10)

|changed| Updated to Python 3.9, with compatibility testing continuing for versions 3.8 and 3.9. Multi-platform CI tests are run on Python 3.9 instead of Python 3.8. CI tests on a Linux machine are also run for versions 3.7 and 3.8. This has been explicitly mentioned in the documentation.

|changed| Updated to Click 8.0.

|changed| Updated CBC Windows binary link in documentation to version 2.10.8.

|fixed| SPORES mode scoring will ignore technologies with energy capacities that are equal to their minimum capacities (i.e., `energy_cap_min`) or which have fixed energy capacities (`energy_cap_equals`).

|fixed| SPORE number is retained when continuing a model run in SPORES mode when solutions already exist for SPORE >= 1. Previously, the SPORE number would be reset to zero.

|fixed| Malformed carrier-specific group constraints are skipped without skipping all subsequent group constraints.

|fixed| Spurious negative values in `storage_inital` in operate mode are ignored in subsequent optimisation runs (#379). Negative values are a result of optimisation tolerances allowing a strictly positive decision variable to end up with (very small in magnitude) negative values. Forcing these to zero between operate mode runs ensures that Pyomo doesn't raise an exception that input values are outside the valid domain (NonNegativeReals).

|fixed| `om_annual` investment costs will be calculated for technologies with only an `om_annual` cost defined in their configuration (#373). Previously, no investment costs would be calculated in this edge case.

## 0.6.8 (2022-02-07)

|new| run configuration parameter to enable relaxation of the `demand_share_per_timestep_decision` constraint.

|new| `storage_cap_min/equals/max` group constraints added.

|changed| Updated to Pyomo 6.2, pandas 1.3, xarray 0.20, numpy 1.20.

|changed| |backwards-incompatible| parameters defaulting to False now default to None, to avoid confusion with zero. To 'switch off' a constraint, a user should now set it to 'null' rather than 'false' in their YAML configuration.

|changed| `INFO` logging level includes logs for dataset cleaning steps before saving to NetCDF and for instantiation of timeseries clustering/resampling (if taking place).

|fixed| `demand_share_per_timestep_decision` constraint set includes all expected (location, technology, carrier) items. In the previous version, not all expected items were captured.

|fixed| Mixed dtype xarray dataset variables, where one dtype is boolean, are converted to float if possible. This overcomes an error whereby the NetCDF file cannot be created due to a mixed dtype variable.

## 0.6.7 (2021-06-29)

|new| `spores` run mode can skip the cost-optimal run, with the user providing initial conditions for `spores_score` and slack system cost.

|new| Support for Pyomo's `gurobi_persistent` solver interface, which enables a more memory- and time-efficient update and re-running of models. A new backend interface has been added to re-build constraints / the objective in the Gurobi persistent solver after updating Pyomo parameters.

|new| A scenario can now be a mix of overrides *and* other scenarios, not just overrides.

|new| `model.backend.rerun()` can work with both `spores` and `plan` run modes (previously only `plan` worked). In the `spores` case, this only works with a built backend that has not been previously run (i.e. `model.run(build_only=True)`), but allows a user to update constraints etc. before running the SPORES method.

|changed| |backwards-incompatible| Carrier-specific group constraints are only allowed in isolation (one constraint in the group).

|changed| If `ensure_feasibility` is set to `True`, `unmet_demand` will always be returned in the model results, even if the model is feasible. Fixes issue #355.

|changed| Updated to Pyomo 6.0, pandas 1.2, xarray 0.17.

|changed| Update CBC Windows binary link in documentation.

|fixed| AttrDict now has a `__name__` attribute, which makes pytest happy.

|fixed| CLI plotting command has been re-enabled. Fixes issue #341.

|fixed| Group constraints are more robust to variations in user inputs. This entails a trade-off whereby some previously accepted user configurations will no longer be possible, since we want to avoid the complexity of processing them.

|fixed| `demand_share_per_timestep_decision` now functions as expected, where it previously did not enforce the per-timestep share after having decided upon it.

|fixed| Various bugs squashed in running operate mode.

|fixed| Handle number of timesteps lower than the horizon length in `operate` mode (#337).

## 0.6.6 (2020-10-08)

|new| `spores` run mode now available, to find Spatially-explicit Practically Optimal REsultS (SPORES)

|new| New group constraints `carrier_con_min`, `carrier_con_max`, `carrier_con_equals` which restrict the total consumed energy of a subgroup of conversion and/or demand technologies.

|new| Add ability to pass timeseries as dataframes in `calliope.Model` instead of only as CSV files.

|new| Pyomo backend interfaces added to get names of all model objects (`get_all_model_attrs`) and to attach custom constraints to the backend model (`add_constraint`).

|changed| Parameters are assigned a domain in Pyomo based on their dtype in `model_data`

|changed| Internal code reorganisation.

|changed| Updated to Pyomo 5.7, pandas 1.1, and xarray 0.16

|fixed| One-way transmission technologies can have `om` costs

|fixed| Silent override of nested dicts when parsing YAML strings

## 0.6.5 (2020-01-14)

|new| New group constraints `energy_cap_equals`, `resource_area_equals`, and  `energy_cap_share_equals` to add the equality constraint to existing `min/max` group constraints.

|new| New group constraints `carrier_prod_min`, `carrier_prod_max`, and  `carrier_prod_equals` which restrict the absolute energy produced by a subgroup of technologies and locations.

|new| Introduced a `storage_discharge_depth` constraint, which allows to set a minimum stored-energy level to be preserved by a storage technology.

|new| New group constraints `net_import_share_min`, `net_import_share_max`, and `net_import_share_equals` which restrict the net imported energy of a certain carrier into subgroups of locations.

|changed| |backwards-incompatible| Group constraints with the prefix `supply_share` are renamed to use the prefix `carrier_prod_share`. This ensures consistent naming for all group constraints.

|changed| Allowed 'energy_cap_min' for transmission technologies.

|changed| Minor additions made to troubleshooting and development documentation.

|changed| |backwards-incompatible| The backend interface to update a parameter value (`Model.backend.update_param()`) has been updated to allow multiple values in a parameter to be updated at once, using a dictionary.

|changed| Allowed `om_con` cost for demand technologies. This is conceived to allow better representing generic international exports as demand sinks with a given revenue (e.g. the average electricity price on a given bidding zone), not restricted to any particular type of technology.

|changed| |backwards-incompatible| `model.backend.rerun()` returns a calliope Model object instead of an xarray Dataset, allowing a user to access calliope Model methods, such as `get_formatted_array`.

|changed| Carrier ratios can be loaded from file, to allow timeseries carrier ratios to be defined, e.g. ``carrier_ratios.carrier_out_2.heat: file=ratios.csv``.

|changed| Objective function options turned into Pyomo parameters. This allows them to update through the `Model.backend.update_param()` functionality.

|changed| All model defaults have been moved to `defaults.yaml`, removing the need for `model.yaml`. A default location, link and group constraint have been added to `defaults.yaml` to validate input model keys.

|changed| |backwards-incompatible| Revised internal logging and warning structure. Less critical warnings during model checks are now logged directly to the INFO log level, which is displayed by default in the CLI, and can be enabled when running in Python by calling `calliope.set_log_verbosity()` without any options. The `calliope.set_log_level` function has been renamed to `calliope.set_log_verbosity` and includes the ability to easily turn on and off the display of solver output.

|changed| All group constraint values are parameters so they can be updated in the backend model

|fixed| Operate mode checks cleaned up to warn less frequently and to not be so aggressive at editing a users model to fit the operate mode requirements.

|fixed| Documentation distinctly renders inline Python, YAML, and shell code snippets.

|fixed| Tech groups are used to filter technologies to which group constraints can be applied. This ensures that transmission and storage technologies are included in cost and energy capacity group constraints. More comprehensive tests have been added accordingly.

|fixed| Models saved to NetCDF now include the fully built internal YAML model and debug data so that `Model.save_commented_model_yaml()` is available after loading a NetCDF model from disk

|fixed| Fix an issue preventing the deprecated `charge_rate` constraint from working in 0.6.4.

|fixed| Fix an issue that prevented 0.6.4 from loading NetCDF models saved with older versions of Calliope. It is still recommended to only load models with the same version of Calliope that they were saved with, as not all functionality will work when mixing versions.

|fixed| |backwards-incompatible| Updated to require pandas 0.25, xarray 0.14, and scikit-learn 0.22, and verified Python 3.8 compatibility. Because of a bugfix in scikit-learn 0.22, models using k-means clustering with a specified random seed may return different clusters from Calliope 0.6.5 on.

## 0.6.4 (2019-05-27)

|new| New model-wide constraint that can be applied to all, or a subset of, locations and technologies in a model, covering:

* `demand_share`, `supply_share`, `demand_share_per_timestep`, `supply_share_per_timestep`, each of which can specify `min`, `max`, and `equals`, as well as `energy_cap_share_min` and `energy_cap_share_max`. These supersede the `group_share` constraints, which are now deprecated and will be removed in v0.7.0.
* `demand_share_per_timestep_decision`, allowing the model to make decisions on the per-timestep shares of carrier demand met from different technologies.
* `cost_max`, `cost_min`, `cost_equals`, `cost_var_max`, `cost_var_min`, `cost_var_equals`, `cost_investment_max`, `cost_investment_min`, `cost_investment_equals`, which allow a user to constrain costs, including those not used in the objective.
* `energy_cap_min`, `energy_cap_max`, `resource_area_min`, `resource_area_max` which allow to constrain installed capacities of groups of technologies in specific locations.

|new| `asynchronous_prod_con` parameter added to the constraints, to allow a user to fix a storage or transmission technology to only be able to produce or consume energy in a given timestep. This ensures that unphysical dissipation of energy cannot occur in these technologies, by activating a binary variable (`prod_con_switch`) in the backend.

|new| Multi-objective optimisation problems can be defined by linear scalarisation of cost classes, using `run.objective_options.cost_class` (e.g. `{'monetary': 1, 'emissions': 0.1}`, which models an emissions price of 0.1 units of currency per unit of emissions)

|new| Storage capacity can be tied to energy capacity with a new `energy_cap_per_storage_cap_equals` constraint.

|new| The ratio of energy capacity and storage capacity can be constrained with a new `energy_cap_per_storage_cap_min` constraint.

|new| Easier way to save an LP file with a ``--save_lp`` command-line option and a ``Model.to_lp`` method

|new| Documentation has a new layout, better search, and is restructured with various content additions, such as a section on troubleshooting.

|new| Documentation for developers has been improved to include an overview of the internal package structure and a guide to contributing code via a pull request.

|changed| |backwards-incompatible| Scenarios in YAML files defined as list of override names, not comma-separated strings: `fusion_scenario: cold_fusion,high_cost` becomes `fusion_scenario: ['cold_fusion', 'high_cost']`. No change to the command-line interface.

|changed| `charge_rate` has been renamed to `energy_cap_per_storage_cap_max`. `charge_rate` will be removed in Calliope 0.7.0.

|changed| Default value of resource_area_max now is ``inf`` instead of ``0``, deactivating the constraint by default.

|changed| Constraint files are auto-loaded in the pyomo backend and applied in the order set by 'ORDER' variables given in each constraint file (such that those constraints which depend on pyomo expressions existing are built after the expressions are built).

|changed| Error on defining a technology in both directions of the same link.

|changed| Any inexistent locations and / or technologies defined in model-wide (group) constraints will be caught and filtered out, raising a warning of their existence in the process.

|changed| Error on required column not existing in CSV is more explicit.

|changed| |backwards-incompatible| Exit code for infeasible problems now is 1 (no success). This is a breaking change when relying on the exit code.

|changed| `get_formatted_array` improved in both speed and memory consumption.

|changed| `model` and `run` configurations are now available as attributes of the Model object, specifically as editable dictionaries which automatically update a YAML string in the `model_data` xarray dataset attribute list (i.e. the information is stored when sending to the solver backend and when saving to and loading from NetCDF file)

|changed| All tests and example models have been updated to solve with Coin-CBC, instead of GLPK. Documentation has been updated to reflect this, and aid in installing CBC (which is not simple for Windows users).

|changed| Additional and improved pre-processing checks and errors for common model mistakes.

|fixed| Total levelised cost of energy considers all costs, but energy generation only from ``supply``, ``supply_plus``, ``conversion``, and ``conversion_plus``.

|fixed| If a space is left between two locations in a link (i.e. `A, B` instead of `A,B`), the space is stripped, instead of leading to the expectation of a location existing with the name ` B`.

|fixed| Timeseries efficiencies can be included in operate mode without failing on preprocessing checks.

|fixed| Name of data variables is retained when accessed through `model.get_formatted_array()`

|fixed| Systemwide constraints work in models without transmission systems.

|fixed| Updated documentation on amendments of abstract base technology groups.

|fixed| Models without time series data fail gracefully.

|fixed| Unknown technology parameters are detected and the user is warned.

|fixed| Loc::techs with empty cost classes (i.e. value == None) are handled by a warning and cost class deletion, instead of messy failure.

## 0.6.3 (2018-10-03)

|new| Addition of ``flows`` plotting function. This shows production and how much they exchange with other locations. It also provides a slider in order to see flows' evolution through time.

|new| ``calliope generate_runs`` in the command line interface can now produce scripts for remote clusters which require SLURM-based submission (``sbatch...``).

|new| |backwards-incompatible| Addition of ``scenarios``, which complement and expand the existing ``overrides`` functionality.  ``overrides`` becomes a top-level key in model configuration, instead of a separate file. The ``calliope run`` command has a new ``--scenario`` option which replaces --override_file, while ``calliope generate_runs`` has a new ``--scenarios`` option which replaces --override_file and takes a semicolon-separated list of scenario names or of group1,group2 combinations. To convert existing overrides to the new approach, simply group them under a top-level ``overrides`` key and import your existing overrides file from the main model configuration file with ``import: ['your_overrides_file.yaml']``.

|new| Addition of ``calliope generate_scenarios`` command to allow automating the construction of scenarios which consist of many combinations of overrides.

|new| Added ``--override_dict`` option to ``calliope run`` and ``calliope generate_runs`` commands

|new| Added solver performance comparison in the docs. CPLEX & Gurobi are, as expected, the best options. If going open-source & free, CBC is much quicker than GLPK!

|new| Calliope is tested and confirmed to run on Python 3.7

|changed| `resource_unit` - available to `supply`, `supply_plus`, and `demand` technologies - can now be defined as 'energy_per_area', 'energy', or 'energy_per_cap'. 'power' has been removed. If 'energy_per_area' then available resource is the resource (CSV or static value) * resource_area, if 'energy_per_cap' it is resource * energy_cap. Default is 'energy', i.e. resource = available_resource.

|changed| Updated to xarray v0.10.8, including updates to timestep aggregation and NetCDF I/O to handle updated xarray functionality.

|changed| Removed ``calliope convert`` command. If you need to convert a 0.5.x model, first use ``calliope convert`` in Calliope 0.6.2 and then upgrade to 0.6.3 or higher.

|changed| Removed comment persistence in AttrDict and the associated API in order to improve compatibility with newer versions of ruamel.yaml

|fixed| Operate mode is more robust, by being explicit about timestep and loc_tech indexing in `storage_initial` preparation and `resource_cap` checks, respectively, instead of assuming an order.

|fixed| When setting `ensure_feasibility`, the resulting `unmet_demand` variable can also be negative, accounting for possible infeasibility when there is unused supply, once all demand has been met (assuming no load shedding abilities). This is particularly pertinent when the `force_resource` constraint is in place.

|fixed| When applying systemwide constraints to transmission technologies, they are no longer silently ignored. Instead, the constraint value is doubled (to account for the constant existence of a pair of technologies to describe one link) and applied to the relevant transmission techs.

|fixed| Permit groups in override files to specify imports of other YAML files

|fixed| If only `interest_rate` is defined within a cost class of a technology, the entire cost class is correctly removed after deleting the `interest_rate` key. This ensures an empty cost key doesn't break things later on. Fixes issue #113.

|fixed| If time clustering with 'storage_inter_cluster' = True, but no storage technologies, the model doesn't break. Fixes issue #142.

## 0.6.2 (2018-06-04)

|new| ``units_max_systemwide`` and ``units_equals_systemwide`` can be applied to an integer/binary constrained technology (capacity limited by ``units`` not ``energy_cap``, or has an associated ``purchase`` (binary) cost). Constraint works similarly to existing ``energy_cap_max_systemwide``, limiting the number of units of a technology that can be purchased across all locations in the model.

|new| |backwards-incompatible| ``primary_carrier`` for `conversion_plus` techs is now split into ``primary_carrier_in`` and ``primary_carrier_out``. Previously, it only accounted for output costs, by separating it, `om_con` and `om_prod` are correctly accounted for. These are required conversion_plus essentials if there's more than one input and output carrier, respectively.

|new| Storage can be set to cyclic using ``run.cyclic_storage``. The last timestep in the series will then be used as the 'previous day' conditions for the first timestep in the series. This also applies to ``storage_inter_cluster``, if clustering. Defaults to False, with intention of defaulting to True in 0.6.3.

|new| On clustering timeseries into representative days, an additional set of decision variables and constraints is generated. This addition allows for tracking stored energy between clusters, by considering storage between every `datestep` of the original (unclustered) timeseries as well as storage variation within a cluster.

|new| CLI now uses the IPython debugger rather than built-in ``pdb``, which provides highlighting, tab completion, and other UI improvements

|new| AttrDict now persists comments when reading from and writing to YAML files, and gains an API to view, add and remove comments on keys

|fixed| Fix CLI error when running a model without transmission technologies

|fixed| Allow plotting for inputs-only models, single location models, and models without location coordinates

|fixed| Fixed negative ``om_con`` costs in conversion and conversion_plus technologies

## 0.6.1 (2018-05-04)

|new| Addition of user-defined datestep clustering, accessed by `clustering_func`: `file=filename.csv:column` in time aggregation config

|new| Added ``layout_updates`` and ``plotly_kwarg_updates`` parameters to plotting functions to override the generated Plotly configuration and layout

|changed| Cost class and sense (maximize/minimize) for objective function may now be specified in run configuration (default remains monetary cost minimization)

|changed| Cleaned up and documented ``Model.save_commented_model_yaml()`` method

|fixed| Fixed error when calling ``--save_plots`` in CLI

|fixed| Minor improvements to warnings

|fixed| Pure dicts can be used to create a ``Model`` instance

|fixed| ``AttrDict.union`` failed on all-empty nested dicts

## 0.6.0 (2018-04-20)

Version 0.6.0 is an almost complete rewrite of most of Calliope's internals. See [New in v0.6.0](https://calliope.readthedocs.io/en/v0.6.0/user/whatsnew.html) for a more detailed description of the many changes.

### Major changes

|changed| |backwards-incompatible| Substantial changes to model configuration format, including more verbose names for most settings, and removal of run configuration files.

|new| |backwards-incompatible| Complete rewrite of Pyomo backend, including new various new and improved functionality to interact with a built model (see :doc:`user/ref_05_to_06`).

|new| Addition of a ``calliope convert`` CLI tool to convert 0.5.x models to 0.6.0.

|new| Experimental ability to link to non-Pyomo backends.

|new| New constraints: ``resource_min_use`` constraint for ``supply`` and ``supply_plus`` techs.

|changed| |backwards-incompatible| Removal of settings and constraints includes ``subset_x``, ``subset_y``, ``s_time``, ``r2``, ``r_scale_to_peak``, ``weight``.

|changed| |backwards-incompatible| ``system_margin`` constraint replaced with ``reserve_margin`` constraint.

|changed| |backwards-incompatible| Removed the ability to load additional custom constraints or objectives.

## 0.5.5 (2018-02-28)

* |fixed| Allow `r_area` to be non-zero if either of `e_cap.max` or `e_cap.equals` is set, not just `e_cap.max`.
* |fixed| Ensure static parameters in resampled timeseries are caught in constraint generation
* |fixed| Fix time masking when set_t.csv contains sub-hourly resolutions

## 0.5.4 (2017-11-10)

### Major changes

* |fixed| `r_area_per_e_cap` and `r_cap_equals_e_cap` constraints have been separated from r_area and r_cap constraints to ensure that user specified `r_area.max` and `r_cap.max` constraints are observed.

* |changed| technologies and location subsets are now communicated with the solver as a combined location:technology subset, to reduce the problem size, by ignoring technologies at locations in which they have not been allowed. This has shown drastic improvements in Pyomo preprocessing time and memory consumption for certain models.

### Other changes

* |fixed| Allow plotting carrier production using `calliope.analysis.plot_carrier_production` if that carrier does not have an associated demand technology (previously would raise an exception).
* |fixed| Define time clustering method (sum/mean) for more constraints that can be time varying. Previously only included `r` and `e_eff`.
* |changed| storage technologies default `s_cap.max` to `inf`, not 0 and are automatically included in the `loc_tech_store` subset. This ensures relevant constraints are not ignored by storage technologies.
* |changed| Some values in the urban scale MILP example were updated to provide results that would show the functionality more clearly
* |changed| technologies have set colours in the urban scale example model, as random colours were often hideous.
* |changed| ruamel.yaml, not ruamel_yaml, is now used for parsing YAML files.
* |fixed| e_cap constraints for unmet_demand technologies are ignored in operational mode. Capacities are fixed for all other technologies, which previously raised an exception, as a fixed infinite capacity is not physically allowable.
* |fixed| stack_weights were strings rather than numeric datatypes on reading NetCDF solution files.

## 0.5.3 (2017-08-22)

### Major changes

* |new| (BETA) Mixed integer linear programming (MILP) capabilities, when using ``purchase`` cost and/or ``units.max/min/equals`` constraints. Integer/Binary decision variables will be applied to the relevant technology-location sets, avoiding unnecessary complexity by describing all technologies with these decision variables.

### Other changes

* |changed| YAML parser is now ruamel_yaml, not pyyaml. This allows scientific notation of numbers in YAML files (#57)
* |fixed| Description of PV technology in urban scale example model now more realistic
* |fixed| Optional ramping constraint no longer uses backward-incompatible definitions (#55)
* |fixed| One-way transmission no longer forces unidirectionality in the wrong direction
* |fixed| Edge case timeseries resource combinations, where infinite resource sneaks into an incompatible constraint, are now flagged with a warning and ignored in that constraint (#61)
* |fixed| e_cap.equals: 0 sets a technology to a capacity of zero, instead of ignoring the constraint (#63)
* |fixed| depreciation_getter now changes with location overrides, instead of just checking the technology level constraints (#64)
* |fixed| Time clustering now functions in models with time-varying costs (#66)
* |changed| Solution now includes time-varying costs (costs_variable)
* |fixed| Saving to NetCDF does not affect in-memory solution (#62)


## 0.5.2 (2017-06-16)

* |changed| Calliope now uses Python 3.6 by default. From Calliope 0.6.0 on, Python 3.6 will likely become the minimum required version.
* |fixed| Fixed a bug in distance calculation if both lat/lon metadata and distances for links were specified.
* |fixed| Fixed a bug in storage constraints when both ``s_cap`` and ``e_cap`` were constrained but no ``c_rate`` was given.
* |fixed| Fixed a bug in the system margin constraint.

## 0.5.1 (2017-06-14)

|new| |backwards-incompatible| Better coordinate definitions in metadata. Location coordinates are now specified by a dictionary with either lat/lon (for geographic coordinates) or x/y (for generic Cartesian coordinates), e.g. ``{lat: 40, lon: -2}`` or ``{x: 0, y: 1}``. For geographic coordinates, the ``map_boundary`` definition for plotting was also updated in accordance. See the built-in example models for details.

|new| Unidirectional transmission links are now possible. See the [documentation on transmission links](https://calliope.readthedocs.io/en/v0.5.1/user/configuration.html#transmission-links).

### Other changes

* |fixed| Missing urban-scale example model files are now included in the distribution
* |fixed| Edge cases in ``conversion_plus`` constraints addressed
* |changed| Documentation improvements


## 0.5.0 (2017-05-04)

### Major changes

|new| Urban-scale example model, major revisions to the documentation to accommodate it, and a new ``calliope.examples`` module to hold multiple example models. In addition, the ``calliope new`` command now accepts a ``--template`` option to select a template other than the default national-scale example model, e.g.: ``calliope new my_urban_model --template=UrbanScale``.

|new| Allow technologies to generate revenue (by specifying negative costs)

|new| Allow technologies to export their carrier directly to outside the system boundary

|new| Allow storage & supply_plus technologies to define a charge rate (c_rate), linking storage capacity (s_cap) with charge/discharge capacity (e_cap) by s_cap * c_rate => e_cap. As such, either s_cap.max & c_rate or e_cap.max & c_rate can be defined for a technology. The smallest of `s_cap.max * c_rate` and `e_cap.max` will be taken if all three are defined.

|changed| |backwards-incompatible| Revised technology definitions and internal definition of sets and subsets, in particular subsets of various technology types. Supply technologies are now split into two types: ``supply`` and ``supply_plus``. Most of the more advanced functionality of the original ``supply`` technology is now contained in ``supply_plus``, making it necessary to update model definitions accordingly. In addition to the existing ``conversion`` technology type, a new more complex ``conversion_plus`` was added.

### Other changes

* |changed| |backwards-incompatible| Creating a ``Model()`` with no arguments now raises a ``ModelError`` rather than returning an instance of the built-in national-scale example model. Use the new ``calliope.examples`` module to access example models.
* |changed| Improvements to the national-scale example model and its tutorial notebook
* |changed| Removed SolutionModel class
* |fixed| Other minor fixes

## 0.4.1 (2017-01-12)

* |new| Allow profiling with the ``--profile`` and ``--profile_filename`` command-line options
* |new| Permit setting random seed with ``random_seed`` in the run configuration
* |changed| Updated installation documentation using conda-forge package
* |fixed| Other minor fixes

## 0.4.0 (2016-12-09)

### Major changes

|new| Added new methods to deal with time resolution: clustering, resampling, and heuristic timestep selection

|changed| |backwards-incompatible| Major change to solution data structure. Model solution is now returned as a single [xarray DataSet](https://docs.xarray.dev/en/v0.8.2/data-structures.html#dataset) instead of multiple pandas DataFrames and Panels. Instead of as a generic HDF5 file, complete solutions can be saved as a NetCDF4 file via xarray's NetCDF functionality.

While the recommended way to save and process model results is by NetCDF4, CSV saving functionality has now been upgraded for more flexibility. Each variable is saved as a separate CSV file with a single value column and as many index columns as required.

|changed| |backwards-incompatible| Model data structures simplified and based on xarray

### Other changes

* |new| Functionality to post-process parallel runs into aggregated NetCDF files in ``calliope.read``
* |changed| Pandas 0.18/0.19 compatibility
* |changed| 1.11 is now the minimum required numpy version. This version makes datetime64 tz-naive by default, thus preventing some odd behavior when displaying time series.
* |changed| Improved logging, status messages, and error reporting
* |fixed| Other minor fixes

## 0.3.7 (2016-03-10)

### Major changes

|changed| Per-location configuration overrides improved. All technology constraints can now be set on a per-location basis, as can costs. This applies to the following settings:

* ``techname.x_map``
* ``techname.constraints.*``
* ``techname.constraints_per_distance.*``
* ``techname.costs.*``

The following settings cannot be overridden on a per-location basis:

* Any other options directly under ``techname``, such as ``techname.parent`` or ``techname.carrier``
* ``techname.costs_per_distance.*``
* ``techname.depreciation.*``

### Other changes

* |fixed| Improved installation instructions
* |fixed| Pyomo 4.2 API compatibility
* |fixed| Other minor fixes

## 0.3.6 (2015-09-23)

* |fixed| Version 0.3.5 changes were not reflected in tutorial

## 0.3.5 (2015-09-18)

### Major changes

|new| New constraint to constrain total (model-wide) installed capacity of a technology (``e_cap.total_max``), in addition to its per-node capacity (``e_cap.max``)

|changed| Removed the ``level`` option for locations. Level is now implicitly derived from the nested structure given by the ``within`` settings. Locations that define no or an empty ``within`` are implicitly at the topmost (0) level.

|changed| |backwards-incompatible| Revised configuration of capacity constraints: ``e_cap_max`` becomes ``e_cap.max``, addition of ``e_cap.min`` and ``e_cap.equals`` (analogous for r_cap, s_cap, rb_cap, r_area). The ``e_cap.equals`` constraint supersedes ``e_cap_max_force`` (analogous for the other constraints). No backwards-compatibility is retained, models must change all constraints to the new formulation. See :ref:`config_reference_constraints` for a complete list of all available constraints. Some additional constraints have name changes:

* ``e_cap_max_scale`` becomes ``e_cap_scale``
* ``rb_cap_follows`` becomes ``rb_cap_follow``, and addition of ``rb_cap_follow_mode``
* ``s_time_max`` becomes ``s_time.max``

|changed| |backwards-incompatible| All optional constraints are now grouped together, under ``constraints.optional``:

* ``constraints.group_fraction.group_fraction`` becomes ``constraints.optional.group_fraction``
* ``constraints.ramping.ramping_rate`` becomes ``constraints.optional.ramping_rate``

### Other changes

* |new| analysis.map_results function to extract solution details from multiple parallel runs
* |new| Various other additions to analysis functionality, particularly in the analysis_utils module
* |new| analysis.get_levelized_cost to get technology and location specific costs
* |new| Allow dynamically loading time mask functions
* |changed| Improved summary table in the model solution: now shows only aggregate information for transmission technologies, also added missing ``s_cap`` column and technology type
* |fixed| Bug causing some total levelized transmission costs to be infinite instead of zero
* |fixed| Bug causing some CSV solution files to be empty

## 0.3.4 (2015-04-27)

* |fixed| Bug in construction and fixed O&M cost calculations in operational mode

## 0.3.3 (2015-04-03)

### Major changes

|changed| In preparation for future enhancements, the ordering of location levels is flipped. The top-level locations at which balancing takes place is now level 0, and may contain level 1 locations. This is a backwards-incompatible change.

|changed| |backwards-incompatible| Refactored time resolution adjustment functionality. Can now give a list of masks in the run configuration which will all be applied, via ``time.masks``, with a base resolution via ``time.resolution`` (or instead, as before, load a resolution series from file via ``time.file``). Renamed the ``time_functions`` submodule to ``time_masks``.

## Other changes

* |new| Models and runs can have a ``name``
* |changed| More verbose ``calliope run``
* |changed| Analysis tools restructured
* |changed| Renamed ``debug.keepfiles`` setting to ``debug.keep_temp_files`` and better documented debug configuration

## 0.3.2 (2015-02-13)

* |new| Run setting ``model_override`` allows specifying the path to a YAML file with overrides for the model configuration, applied at model initialization (path is given relative to the run configuration file used). This is in addition to the existing ``override`` setting, and is applied first (so ``override`` can override ``model_override``).
* |new| Run settings ``output.save_constraints`` and ``output.save_constraints_options``
* |new| Run setting ``parallel.post_run``
* |changed| Solution column names more in line with model component names
* |changed| Can specify more than one output format as a list, e.g. ``output.format: ['csv', 'hdf']``
* |changed| Run setting ``parallel.additional_lines`` renamed to ``parallel.pre_run``
* |changed| Better error messages and CLI error handling
* |fixed| Bug on saving YAML files with numpy dtypes fixed
* Other minor improvements and fixes

## 0.3.1 (2015-01-06)

* Fixes to time_functions
* Other minor improvements and fixes

## 0.3.0 (2014-12-12)

* Python 3 and Pyomo 4 are now minimum requirements
* Significantly improved documentation
* Improved model solution management by saving to HDF5 instead of CSV
* Calculate shares of technologies, including the ability to define groups for the purpose of computing shares
* Improved operational mode
* Simplified time_tools
* Improved output plotting, including dispatch, transmission flows, and installed capacities, and added model configuration to support these plots
* ``r`` can be specified as power or energy
* Improved solution speed
* Better error messages and basic logging
* Better sanity checking and error messages for common mistakes
* Basic distance-dependent constraints (only implemented for e_loss and cost of e_cap for now)
* Other improvements and fixes

## 0.2.0 (2014-03-18)

* Added cost classes with a new set ``k``
* Added energy carriers with a new set ``c``
* Added conversion technologies
* Speed improvements and simplifications
* Ability to arbitrarily nest model configuration files with ``import`` statements
* Added additional constraints
* Improved configuration handling
* Ability to define timestep options in run configuration
* Cleared up terminology (nodes vs locations)
* Improved TimeSummarizer masking and added new masks
* Removed technology classes
* Improved operational mode with results output matching planning mode and dynamic updating of parameters in model instance
* Working parallel_tools
* Improved documentation
* Apache 2.0 licensed
* Other improvements and fixes

## 0.1.0 (2013-12-10)

* Some semblance of documentation
* Usable built-in example model
* Improved and working TimeSummarizer
* More flexible masking for TimeSummarizer
* Ability to add additional constraints without editing core source code
* Some basic test coverage
* Working parallel run configuration system
