----------------------
Advanced functionality
----------------------


.. _configuration_timeseries:

Using time series data
----------------------

.. Note::

   If a parameter is not explicit in time and space, it can be specified as a single value in the model definition (or, using location-specific overrides, be made spatially explicit). This applies both to parameters that never vary through time (for example, cost of installed capacity) and for those that may be time-varying (for example, a technology's available resource).

Defining a model's time steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Irrespective of whether it actually uses time-varying parameters, a model must at specify its timesteps with a file called ``set_t.csv``. This must contain two columns (comma-separated), the first one being integer indices, and the second, ISO 8601 compatible timestamps (usually in the format ``YYYY-MM-DD hh:mm:ss``, e.g. ``2005-01-01 00:00:00``).

For example, the first few lines of a file specifying hourly timesteps for the year 2005 would look like this:

.. code-block:: text

   0,2005-01-01 00:00:00
   1,2005-01-01 01:00:00
   2,2005-01-01 02:00:00
   3,2005-01-01 03:00:00
   4,2005-01-01 04:00:00
   5,2005-01-01 05:00:00
   6,2005-01-01 06:00:00

Defining time-varying parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For parameters that vary in time, time series data can be read from CSV files. This can be done in two ways (using the example of ``r``):

1. Specify ``r: file=filename.csv`` to pick the desired CSV file.
2. Specify ``r: file``. In this case, the file name is automatically determined according to the format ``tech_param.csv`` (e.g., ``pv_r.csv`` for the parameter ``r`` of a technology with the identifier ``pv``).

Each CSV file must have integer indices in the first column which match the integer indices from ``set_t.csv``. The first row must be column names, while the rest of the cells are the actual (integer or floating point) data values:

.. code-block:: text

   ,loc1,loc2,loc3,...
   0,10,20,10.0,...
   1,11,19,9.9,...
   2,12,18,9.8,...
   ...

In the most straightforward case, the column names in the CSV files correspond to the location names defined in the model (in the above example, ``loc1``, ``loc2`` and ``loc3``). However, it is possible to define a mapping of column names to locations. For example, if our model has two locations, ``uk`` and ``germany``, but the electricity demand data columns are ``loc1``, ``loc2`` and ``loc3``, then the following ``x_map`` definition will read the demand data for the desired locations from the specified columns:

.. code-block:: yaml

   electricity_demand:
      x_map: 'uk: loc1, germany: loc2'
      constraints:
         r: 'file=demand.csv'

.. Warning::

   After reading a CSV file, if any columns are missing (i.e. if a file does not contain columns for all locations defined in the current model), the value for those locations is simply set to :math:`0` for all timesteps.

.. Note::

   ``x_map`` maps column names in an input CSV file to locations defined in the model, in the format ``name_in_model: name_in_file``, with as many comma-separated such definitions as necessary.

In all cases, all CSV files, alongside ``set_t.csv``, must be inside the data directory specified by ``data_path`` in the model definition.

For example, the files for a model specified in ``model.yaml``, which defined ``data_path: model_data``, might look like this (``+`` are directories, ``-`` files):

.. code-block:: text

   - model.yaml
   + model_data/
      - set_t.csv
      - tech1_r.csv
      - tech2_r.csv
      - tech2_e_eff.csv
      - ...

When reading time series, the ``r_scale_to_peak`` option can be useful. Specifying this will automatically scale the time series so that the peak matches the given value. In the case of ``r`` for demand technologies, where ``r`` will be negative, the peak is instead a trough, and this is handled automatically. In the below example, the electricity demand timeseries is loaded from ``demand.csv`` and scaled such that the demand peak is 60,000:

.. code-block:: yaml

   electricity_demand:
      constraints:
         r: 'file=demand.csv'
         r_scale_to_peak: -60000


.. _time_clustering:

Time resolution adjustment
--------------------------

Models must have a default timestep length (defined implicitly by the timesteps defined in ``set_t.csv``), and all time series files used in a given model must conform to that timestep length requirement.

However, this default resolution can be adjusted over parts of the dataset via configuring ``time`` in the run settings. At its most basic, this allows running a function that can perform arbitrary adjustments to the time series data in the model, via ``time.function``, and/or applying a series of masks to the time series data to select specific areas of interest, such as periods of maximum or minimum production by certain technologies, via ``time.masks``.

The available options include:

1. Uniform time resolution reduction through the resample function, which takes a `pandas-compatible rule describing the target resolution <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html>`_. For example, to resample to 6-hourly timesteps:

.. code-block:: yaml

   time:
       function: resample
       function_options: {'resolution': '6H'}

2. Deriving representative days from the input time series, by applying either k-means or hierarchical clustering as defined in :mod:`calliope.time_clustering`, for example:

.. code-block:: yaml

   time:
       function: apply_clustering
       function_options: {clustering_func: 'get_clusters_kmeans', how: 'mean', k: 20}

3. Heuristic selection: application of one or more of the masks defined in :mod:`calliope.time_masks`, via a list of masks given in ``time.masks``. See :ref:`api_time_masks` in the API documentation for the available masking functions. Options can be passed to the masking functions by specifying ``options``. A ``time.function`` can still be specified and will be applied to the masked areas (i.e. those areas of the time series not selected), as in this example which looks for the week of minimum and maximum potential wind production (assuming a ``wind`` technology was specified), then reduces the rest of the input time series to 6-hourly resolution:

.. code-block:: yaml

   time:
      masks:
          - {function: week, options: {day_func: 'extreme', tech: 'wind', how: 'max'}}
          - {function: week, options: {day_func: 'extreme', tech: 'wind', how: 'min'}}
      function: resample
      function_options: {'resolution': '6H'}

.. Note::

  When loading a model, all time steps initially have the same weight. Time step resolution reduction methods may adjust the weight of individual timesteps; this is used for example to give appropriate weight to the operational costs of aggregated typical days in comparison to individual extreme days, if both exist in the same processed time series. See the implementation of constraints in :mod:`calliope.constraints.base` for more detail.


.. _operational_mode:

Operational mode
----------------

Requires two model settings:

.. code-block:: yaml

    model:
        operation:
            horizon:
            window:

.. _run_config_generate:

Overrides and generating scripts for running multiple scenarios
---------------------------------------------------------------

* Create a ``run.yaml`` file with a ``parallel:`` section as needed (see :ref:`run_config_generate`).
* On the command line, run ``calliope generate path/to/run.yaml``.
* By default, this will create a new subdirectory inside a ``runs`` directory in the current working directory. You can optionally specify a different target directory by passing a second path to ``calliope generate``, e.g. ``calliope generate path/to/run.yaml path/to/my_run_files``.
* Calliope generates several files and directories in the target path. The most important are the ``Runs`` subdirectory which hosts the self-contained configuration for the runs and ``run.sh`` script, which is responsible for executing each run. In order to execute these runs in parallel on a compute cluster, a submit.sh script is also generated containing job control data, and which can be submitted via a cluster controller (e.g., ``qsub submit.sh``).

The ``run.sh`` script can simply be called with an integer argument from the sequence (1, number of parallel runs) to execute a given run, e.g. ``run.sh 1``, ``run.sh 2``, etc. This way the runs can easily be executed irrespective of the parallel computing environment available.

.. Note:: Models generated via ``calliope generate`` automatically save results as a single NetCDF file per run inside the parallel runs' ``Output`` subdirectory, regardless of whether the ``output.path`` or ``output.format`` options have been set.

The run settings can also include a ``parallel`` section.

This section is parsed when using the ``calliope generate`` command-line tool to generate a set of runs to be executed in parallel. A run settings file defining ``parallel`` can still be used to execute a single model run, in which case the ``parallel`` section is simply ignored.

The concept behind parallel runs is to specify a base model (via the run configuration's ``model`` setting), then define a set of model runs using this base model, but overriding one or a small number of settings in each run. For example, one could explore a range of costs of a specific technology and how this affects the result.

Specifying these iterations is not (yet) automated, they must be manually entered under ``parallel.iterations:`` section. However, Calliope provides functionality to gather and process the results from a set of parallel runs (see :doc:`analysing`).

At a minimum, the ``parallel`` block must define:

* a ``name`` for the run
* the ``environment`` of the cluster (if it is to be run on a cluster), currently supported is ``bsub`` and ``qsub``. In either case, the generated scripts can also be run manually
* ``iterations``: a list of model runs, with each entry giving the settings that should be overridden for that run. The settings are *run settings*, so, for example, ``time.function`` can be overridden. Because the run settings can themselves override model settings, via ``override``, model settings can be specified here, e.g. ``override.techs.nuclear.costs.monetary.e_cap``.

The following example parallel settings show the available options. In this case, two iterations are defined, and each of them overrides the nuclear ``e_cap`` costs (``override.techs.nuclear.costs.monetary.e_cap``):

.. code-block:: yaml

   parallel:
       name: 'example-model'  # Name of this run
       environment: 'bsub'  # Cluster environment, choices: bsub, qsub
       data_path_adjustment: '../../../model_config'
       # Execute additional commands in the run script before starting the model
       pre_run: ['source activate pyomo']
       # Execute additional commands after running the model
       post_run: []
       iterations:
           - override.techs.nuclear.costs.monetary.e_cap: 1000
           - override.techs.nuclear.costs.monetary.e_cap: 2000
       resources:
           threads: 1  # Set to request a non-default number of threads
           wall_time: 30  # Set to request a non-default run time in minutes
           memory: 1000  # Set to request a non-default amount of memory in MB

This also shows the optional settings available:

* ``data_path_adjustment``: replaces the ``data_path`` setting in the model configuration during parallel runs only
* ``pre_run`` and ``post_run``: one or multiple lines (given as a list) that will be executed in the run script before / after running the model. If running on a computing cluster, ``pre_run`` is likely to include a line or two setting up any environment variables and activating the necessary Python environment.
* ``resources``: specifying these will include resource requests to the cluster controller into the generated run scripts. ``threads``, ``wall_time``, and ``memory`` are available. Whether and how these actually get processed or honored depends on the setup of the cluster environment.

For an iteration to override more than one setting at a time, the notation is as follows:

.. code-block:: yaml

   iterations:
       - first_option: 500
         second_option: 10
       - first_option: 600
         second_option: 20


.. _overriding_tech_options:

Overriding technology options
-----------------------------

Technologies can define generic options, for example ``name``, constraints, for example ``constraints.e_cap.max``, and costs, for example ``costs.monetary.e_cap``.

These options can be overridden in several ways, and whenever such an option is accessed by Calliope it works its way through the following list until it finds a definition (so entries further up in this list take precedence over those further down):

1. Override for a specific location ``x1`` and technology ``y1``, which may be defined via ``locations`` (e.g. ``locations.x1.override.y1.constraints.e_cap.max``)
2. Setting specific to the technology ``y1`` if defined in ``techs`` (e.g. ``techs.y1.constraints.e_cap.max``)
3. Check whether the immediate parent of the technology ``y`` defines the option (assuming that ``y1`` specifies ``parent: my_parent_tech``, e.g. ``techs.my_parent_tech.constraints.e_cap.max``)
4. If the option is still not found, continue along the chain of parent-child relationships. Since every technology should inherit from one of the abstract base technologies, and those in turn inherit from the model-wide defaults, this will ultimately lead to the model-wide default setting if it has not been specified anywhere else. See :ref:`config_reference_constraints` for a complete listing of those defaults.

The following technology options can be overriden on a per-location basis:

* ``x_map``
* ``constraints.*``
* ``constraints_per_distance.*``
* ``costs.*``

The following settings cannot be overridden on a per-location basis:

* Any other options, such as ``parent`` or ``carrier``
* ``costs_per_distance.*``
* ``depreciation.*``

Binary and mixed-integer models
-------------------------------

By applying a ``purchase`` cost to a technology, that technology will have a binary variable associated with it, describing whether or not it has been "purchased".

By applying ``units.max``, ``units.min``, or ``units.equals`` to a technology, that technology will have a integer variable associated with it, describing how many of that technology have been "purchased". If a ``purchase`` cost has been applied to this same technology, the purchasing cost will be applied per unit.

.. Warning::

   Integer and Binary variables are still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

.. seealso:: :ref:`milp_example_model`
