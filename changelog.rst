.. include:: definitions.rst

Release History
===============

0.6.4 (dev)
-----------

|fixed| Updated documentation on amendments of abstract base technology groups

|fixed| Models without time series data fail gracefully.

0.6.3 (2018-10-03)
------------------

|new| Addition of ``flows`` plotting function. This shows production and how much they exchange with other locations. It also provides a slider in order to see flows' evolution through time.

|new| ``calliope generate_runs`` in the command line interface can now produce scripts for remote clusters which require SLURM-based submission (``sbatch...``).

|new| |backwards-incompatible| Addition of ``scenarios``, which complement and expand the existing ``overrides`` functionality.  ``overrides`` becomes a top-level key in model configuration, instead of a separate file. The ``calliope run`` command has a new ``--scenario`` option which replaces --override_file, while ``calliope generate_runs`` has a new ``--scenarios`` option which replaces --override_file and takes a semicolon-separated list of scenario names or of group1,group2 combinations. To convert existing overrides to the new approach, simply group them under a top-level ``overrides`` key and import your existing overrides file from the main model configuration file with ``import: ['your_overrides_file.yaml']``.

|new| Addition of ``calliope generate_scenarios`` command to allow building autmating the construction of scenarios which consist of many combinations of overrides.

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

0.6.2 (2018-06-04)
------------------

|new| ``units_max_systemwide`` and ``units_equals_systemwide`` can be applied to an integer/binary constrained technology (capacity limited by ``units`` not ``energy_cap``, or has an associated ``purchase`` (binary) cost). Constraint works similarly to existing ``energy_cap_max_systemwide``, limiting the number of units of a technology that can be purchased across all locations in the model.

|new| |backwards-incompatible| ``primary_carrier`` for `conversion_plus` techs is now split into ``primary_carrier_in`` and ``primary_carrier_out``. Previously, it only accounted for output costs, by separating it, `om_con` and `om_prod` are correctly accounted for. These are required conversion_plus essentials if there's more than one input and output carrier, respectively.

|new| Storage can be set to cyclic using ``run.cyclic_storage``. The last timestep in the series will then be used as the 'previous day' conditions for the first timestep in the series. This also applies to ``storage_inter_cluster``, if clustering. Defaults to False, with intention of defaulting to True in 0.6.3.

|new| On clustering timeseries into representative days, an additional set of decision variables and constraints is generated. This addition allows for tracking stored energy between clusters, by considering storage between every `datestep` of the original (unclustered) timeseries as well as storage variation within a cluster.

|new| CLI now uses the IPython debugger rather than built-in ``pdb``, which provides highlighting, tab completion, and other UI improvements

|new| AttrDict now persists comments when reading from and writing to YAML files, and gains an API to view, add and remove comments on keys

|fixed| Fix CLI error when running a model without transmission technologies

|fixed| Allow plotting for inputs-only models, single location models, and models without location coordinates

|fixed| Fixed negative ``om_con`` costs in conversion and conversion_plus technologies

0.6.1 (2018-05-04)
------------------

|new| Addition of user-defined datestep clustering, accessed by `clustering_func`:`file=filename.csv:column` in time aggregation config

|new| Added ``layout_updates`` and ``plotly_kwarg_updates`` parameters to plotting functions to override the generated Plotly configuration and layout

|changed| Cost class and sense (maximize/minimize) for objective function may now be specified in run configuration (default remains monetary cost minimization)

|changed| Cleaned up and documented ``Model.save_commented_model_yaml()`` method

|fixed| Fixed error when calling ``--save_plots`` in CLI

|fixed| Minor improvements to warnings

|fixed| Pure dicts can be used to create a ``Model`` instance

|fixed| ``AttrDict.union`` failed on all-empty nested dicts

0.6.0 (2018-04-20)
------------------

Version 0.6.0 is an almost complete rewrite of most of Calliope's internals. See :doc:`user/whatsnew_060` for a more detailed description of the many changes.

Major changes
~~~~~~~~~~~~~

|changed| |backwards-incompatible| Substantial changes to model configuration format, including more verbose names for most settings, and removal of run configuration files.

|new| |backwards-incompatible| Complete rewrite of Pyomo backend, including new various new and improved functionality to interact with a built model (see :doc:`user/whatsnew_060`).

|new| Addition of a ``calliope convert`` CLI tool to convert 0.5.x models to 0.6.0.

|new| Experimental ability to link to non-Pyomo backends.

|new| New constraints: ``resource_min_use`` constraint for ``supply`` and ``supply_plus`` techs.

|changed| |backwards-incompatible| Removal of settings and constraints includes ``subset_x``, ``subset_y``, ``s_time``, ``r2``, ``r_scale_to_peak``, ``weight``.

|changed| |backwards-incompatible| ``system_margin`` constraint replaced with ``reserve_margin`` constraint.

|changed| |backwards-incompatible| Removed the ability to load additional custom constraints or objectives.

0.5.5 (2018-02-28)
------------------

* |fixed| Allow `r_area` to be non-zero if either of `e_cap.max` or `e_cap.equals` is set, not just `e_cap.max`.
* |fixed| Ensure static parameters in resampled timeseries are caught in constraint generation
* |fixed| Fix time masking when set_t.csv contains sub-hourly resolutions

0.5.4 (2017-11-10)
------------------

Major changes
~~~~~~~~~~~~~
* |fixed| `r_area_per_e_cap` and `r_cap_equals_e_cap` constraints have been separated from r_area and r_cap constraints to ensure that user specified `r_area.max` and `r_cap.max` constraints are observed.

* |changed| technologies and location subsets are now communicated with the solver as a combined location:technology subset, to reduce the problem size, by ignoring technologies at locations in which they have not been allowed. This has shown drastic improvements in Pyomo preprocessing time and memory consumption for certain models.

Other changes
~~~~~~~~~~~~~

* |fixed| Allow plotting carrier production using `calliope.analysis.plot_carrier_production` if that carrier does not have an associated demand technology (previously would raise an exception).
* |fixed| Define time clustering method (sum/mean) for more constraints that can be time varying. Previously only included `r` and `e_eff`.
* |changed| storage technologies default `s_cap.max` to `inf`, not 0 and are automatically included in the `loc_tech_store` subset. This ensures relevant constraints are not ignored by storage technologies.
* |changed| Some values in the urban scale MILP example were updated to provide results that would show the functionality more clearly
* |changed| technologies have set colours in the urban scale example model, as random colours were often hideous.
* |changed| ruamel.yaml, not ruamel_yaml, is now used for parsing YAML files.
* |fixed| e_cap constraints for unmet_demand technologies are ignored in operational mode. Capacities are fixed for all other technologies, which previously raised an exception, as a fixed infinite capacity is not physically allowable.
* |fixed| stack_weights were strings rather than numeric datatypes on reading NetCDF solution files.

0.5.3 (2017-08-22)
------------------

Major changes
~~~~~~~~~~~~~

* |new| (BETA) Mixed integer linear programming (MILP) capabilities, when using ``purchase`` cost and/or ``units.max/min/equals`` constraints. Integer/Binary decision variables will be applied to the relevant technology-location sets, avoiding unnecessary complexity by describing all technologies with these decision variables.

Other changes
~~~~~~~~~~~~~

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

0.5.2 (2017-06-16)
------------------

* |changed| Calliope now uses Python 3.6 by default. From Calliope 0.6.0 on, Python 3.6 will likely become the minimum required version.
* |fixed| Fixed a bug in distance calculation if both lat/lon metadata and distances for links were specified.
* |fixed| Fixed a bug in storage constraints when both ``s_cap`` and ``e_cap`` were constrained but no ``c_rate`` was given.
* |fixed| Fixed a bug in the system margin constraint.

0.5.1 (2017-06-14)
------------------

|new| |backwards-incompatible| Better coordinate definitions in metadata. Location coordinates are now specified by a dictionary with either lat/lon (for geographic coordinates) or x/y (for generic Cartesian coordinates), e.g. ``{lat: 40, lon: -2}`` or ``{x: 0, y: 1}``. For geographic coordinates, the ``map_boundary`` definition for plotting was also updated in accordance. See the built-in example models for details.

|new| Unidirectional transmission links are now possible. See the `documentation on transmission links <https://calliope.readthedocs.io/en/stable/user/configuration.html#transmission-links>`_.

Other changes
~~~~~~~~~~~~~

* |fixed| Missing urban-scale example model files are now included in the distribution
* |fixed| Edge cases in ``conversion_plus`` constraints addressed
* |changed| Documentation improvements

0.5.0 (2017-05-04)
------------------

Major changes
~~~~~~~~~~~~~

|new| Urban-scale example model, major revisions to the documentation to accommodate it, and a new ``calliope.examples`` module to hold multiple example models. In addition, the ``calliope new`` command now accepts a ``--template`` option to select a template other than the default national-scale example model, e.g.: ``calliope new my_urban_model --template=UrbanScale``.

|new| Allow technologies to generate revenue (by specifying negative costs)

|new| Allow technologies to export their carrier directly to outside the system boundary

|new| Allow storage & supply_plus technologies to define a charge rate (c_rate), linking storage capacity (s_cap) with charge/discharge capacity (e_cap) by s_cap * c_rate => e_cap. As such, either s_cap.max & c_rate or e_cap.max & c_rate can be defined for a technology. The smallest of `s_cap.max * c_rate` and `e_cap.max` will be taken if all three are defined.

|changed| |backwards-incompatible| Revised technology definitions and internal definition of sets and subsets, in particular subsets of various technology types. Supply technologies are now split into two types: ``supply`` and ``supply_plus``. Most of the more advanced functionality of the original ``supply`` technology is now contained in ``supply_plus``, making it necessary to update model definitions accordingly. In addition to the existing ``conversion`` technology type, a new more complex ``conversion_plus`` was added.

Other changes
~~~~~~~~~~~~~

* |changed| |backwards-incompatible| Creating a ``Model()`` with no arguments now raises a ``ModelError`` rather than returning an instance of the built-in national-scale example model. Use the new ``calliope.examples`` module to access example models.
* |changed| Improvements to the national-scale example model and its tutorial notebook
* |changed| Removed SolutionModel class
* |fixed| Other minor fixes

0.4.1 (2017-01-12)
------------------

* |new| Allow profiling with the ``--profile`` and ``--profile_filename`` command-line options
* |new| Permit setting random seed with ``random_seed`` in the run configuration
* |changed| Updated installation documentation using conda-forge package
* |fixed| Other minor fixes

0.4.0 (2016-12-09)
------------------

Major changes
~~~~~~~~~~~~~

|new| Added new methods to deal with time resolution: clustering, resampling, and heuristic timestep selection

|changed| |backwards-incompatible| Major change to solution data structure. Model solution is now returned as a single `xarray DataSet <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ instead of multiple pandas DataFrames and Panels. Instead of as a generic HDF5 file, complete solutions can be saved as a NetCDF4 file via xarray's NetCDF functionality.

While the recommended way to save and process model results is by NetCDF4, CSV saving functionality has now been upgraded for more flexibility. Each variable is saved as a separate CSV file with a single value column and as many index columns as required.

|changed| |backwards-incompatible| Model data structures simplified and based on xarray

Other changes
~~~~~~~~~~~~~

* |new| Functionality to post-process parallel runs into aggregated NetCDF files in ``calliope.read``
* |changed| Pandas 0.18/0.19 compatibility
* |changed| 1.11 is now the minimum required numpy version. This version makes datetime64 tz-naive by default, thus preventing some odd behavior when displaying time series.
* |changed| Improved logging, status messages, and error reporting
* |fixed| Other minor fixes

0.3.7 (2016-03-10)
------------------

Major changes
~~~~~~~~~~~~~

|changed| Per-location configuration overrides improved. All technology constraints can now be set on a per-location basis, as can costs. This applies to the following settings:

* ``techname.x_map``
* ``techname.constraints.*``
* ``techname.constraints_per_distance.*``
* ``techname.costs.*``

The following settings cannot be overridden on a per-location basis:

* Any other options directly under ``techname``, such as ``techname.parent`` or ``techname.carrier``
* ``techname.costs_per_distance.*``
* ``techname.depreciation.*``

Other changes
~~~~~~~~~~~~~

* |fixed| Improved installation instructions
* |fixed| Pyomo 4.2 API compatibility
* |fixed| Other minor fixes

0.3.6 (2015-09-23)
------------------

* |fixed| Version 0.3.5 changes were not reflected in tutorial

0.3.5 (2015-09-18)
------------------

Major changes
~~~~~~~~~~~~~

|new| New constraint to constrain total (model-wide) installed capacity of a technology (``e_cap.total_max``), in addition to its per-node capacity (``e_cap.max``)

|changed| Removed the ``level`` option for locations. Level is now implicitly derived from the nested structure given by the ``within`` settings. Locations that define no or an empty ``within`` are implicitly at the topmost (0) level.

|changed| |backwards-incompatible| Revised configuration of capacity constraints: ``e_cap_max`` becomes ``e_cap.max``, addition of ``e_cap.min`` and ``e_cap.equals`` (analogous for r_cap, s_cap, rb_cap, r_area). The ``e_cap.equals`` constraint supersedes ``e_cap_max_force`` (analogous for the other constraints). No backwards-compatibility is retained, models must change all constraints to the new formulation. See :ref:`config_reference_constraints` for a complete list of all available constraints. Some additional constraints have name changes:

* ``e_cap_max_scale`` becomes ``e_cap_scale``
* ``rb_cap_follows`` becomes ``rb_cap_follow``, and addition of ``rb_cap_follow_mode``
* ``s_time_max`` becomes ``s_time.max``

|changed| |backwards-incompatible| All optional constraints are now grouped together, under ``constraints.optional``:

* ``constraints.group_fraction.group_fraction`` becomes ``constraints.optional.group_fraction``
* ``constraints.ramping.ramping_rate`` becomes ``constraints.optional.ramping_rate``

Other changes
~~~~~~~~~~~~~

* |new| analysis.map_results function to extract solution details from multiple parallel runs
* |new| Various other additions to analysis functionality, particularly in the analysis_utils module
* |new| analysis.get_levelized_cost to get technology and location specific costs
* |new| Allow dynamically loading time mask functions
* |changed| Improved summary table in the model solution: now shows only aggregate information for transmission technologies, also added missing ``s_cap`` column and technology type
* |fixed| Bug causing some total levelized transmission costs to be infinite instead of zero
* |fixed| Bug causing some CSV solution files to be empty

0.3.4 (2015-04-27)
------------------

* |fixed| Bug in construction and fixed O&M cost calculations in operational mode

0.3.3 (2015-04-03)
------------------

Major changes
~~~~~~~~~~~~~

|changed| In preparation for future enhancements, the ordering of location levels is flipped. The top-level locations at which balancing takes place is now level 0, and may contain level 1 locations. This is a backwards-incompatible change.

|changed| |backwards-incompatible| Refactored time resolution adjustment functionality. Can now give a list of masks in the run configuration which will all be applied, via ``time.masks``, with a base resolution via ``time.resolution`` (or instead, as before, load a resolution series from file via ``time.file``). Renamed the ``time_functions`` submodule to ``time_masks``.

Other changes
~~~~~~~~~~~~~

* |new| Models and runs can have a ``name``
* |changed| More verbose ``calliope run``
* |changed| Analysis tools restructured
* |changed| Renamed ``debug.keepfiles`` setting to ``debug.keep_temp_files`` and better documented debug configuration

0.3.2 (2015-02-13)
------------------

* |new| Run setting ``model_override`` allows specifying the path to a YAML file with overrides for the model configuration, applied at model initialization (path is given relative to the run configuration file used). This is in addition to the existing ``override`` setting, and is applied first (so ``override`` can override ``model_override``).
* |new| Run settings ``output.save_constraints`` and ``output.save_constraints_options``
* |new| Run setting ``parallel.post_run``
* |changed| Solution column names more in line with model component names
* |changed| Can specify more than one output format as a list, e.g. ``output.format: ['csv', 'hdf']``
* |changed| Run setting ``parallel.additional_lines`` renamed to ``parallel.pre_run``
* |changed| Better error messages and CLI error handling
* |fixed| Bug on saving YAML files with numpy dtypes fixed
* Other minor improvements and fixes

0.3.1 (2015-01-06)
------------------

* Fixes to time_functions
* Other minor improvements and fixes

0.3.0 (2014-12-12)
------------------

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

0.2.0 (2014-03-18)
------------------

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

0.1.0 (2013-12-10)
------------------

* Some semblance of documentation
* Usable built-in example model
* Improved and working TimeSummarizer
* More flexible masking for TimeSummarizer
* Ability to add additional constraints without editing core source code
* Some basic test coverage
* Working parallel run configuration system
