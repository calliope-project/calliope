.. include:: definitions.rst

Release History
===============

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
