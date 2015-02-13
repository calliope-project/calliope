
Release History
---------------

0.3.2 (2015-02-13)
++++++++++++++++++

* [new] Run setting ``model_override`` allows specifying the path to a YAML file with overrides for the model configuration, applied at model initialization (path is given relative to the run configuration file used). This is in addition to the existing ``override`` setting, and is applied first (so ``override`` can override ``model_override``).
* [new] Run settings ``output.save_constraints`` and ``output.save_constraints_options``
* [new] Run setting ``parallel.post_run``
* [changed] Solution column names more in line with model component names
* [changed] Can specify more than one output format as a list, e.g. ``output.format: ['csv', 'hdf']``
* [changed] Run setting ``parallel.additional_lines`` renamed to ``parallel.pre_run``
* [changed] Better error messages and CLI error handling
* [fixed] Bug on saving YAML files with numpy dtypes fixed
* [changed/fixed] Other minor improvements and fixes

0.3.1 (2015-01-06)
++++++++++++++++++

* Fixes to time_functions
* Other minor improvements and fixes

0.3.0 (2014-12-12)
++++++++++++++++++

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
++++++++++++++++++

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
++++++++++++++++++

* Some semblance of documentation
* Usable built-in example model
* Improved and working TimeSummarizer
* More flexible masking for TimeSummarizer
* Ability to add additional constraints without editing core source code
* Some basic test coverage
* Working parallel run configuration system
