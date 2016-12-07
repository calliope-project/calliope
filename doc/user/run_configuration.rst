
=================
Run configuration
=================

.. Note::

   See :ref:`config_reference_run` in the configuration reference for a complete listing of all available configuration options.

At a minimum, the run configuration must provide three settings, as shown in this example:

.. code-block:: yaml

   model: 'model_config/model.yaml'
   mode: 'plan'
   solver: 'glpk'

``model`` specifies the path to the model configuration file for the model to be run. ``mode`` specifies whether the model should be run in planning (``plan``) or operational (``operate``) mode (see :doc:`running`). Finally, ``solver`` specifies the solver to be used. Calliope has been tested with GLPK, Gurobi and CPLEX. Any of the solvers that Pyomo is compatible with should work.

Additional (optional) settings, including debug settings, can be specified in the run configuration. In particular, the run settings can override any model settings by specifying ``override``, e.g.:

.. code-block:: yaml

   override:
      techs:
         nuclear:
            costs:
               monetary:
                  e_cap: 1000

.. Note:: If run settings override the ``data_path`` setting and specify a relative path, that path will be interpreted as relative to the run settings file and not the model settings file being overridden.

.. TODO add documentation on special _REPLACE_ key

Instead of directly overriding settings within the run configuration file using an ``override`` block, it is also possible to specify an additional model configuration file with overriding settings by using the ``model_override: path/to/model_override.yaml`` setting (the path given here is relative to the run configuration file).

The optional settings to adjust the timestep resolution and those for parallel runs are discussed below. For a complete list of the other available settings, see :ref:`config_reference_run` in the configuration reference.

.. _run_time_res:

--------------------------
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

.. _run_config_parallel_runs:

--------------------------
Settings for parallel runs
--------------------------

The run settings can also include a ``parallel`` section.

This section is parsed when using the ``calliope generate`` command-line tool to generate a set of runs to be executed in parallel (see :ref:`parallel_runs`). A run settings file defining ``parallel`` can still be used to execute a single model run, in which case the ``parallel`` section is simply ignored.

The concept behind parallel runs is to specify a base model (via the run configuration's ``model`` setting), then define a set of model runs using this base model, but overriding one or a small number of settings in each run. For example, one could explore a range of costs of a specific technology and how this affects the result.

Specifying these iterations is not (yet) automated, they must be manually entered under ``parallel.iterations:`` section. However, Calliope provides functionality to gather and process the results from a set of parallel runs (see :doc:`analysis`).

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

See :ref:`parallel_runs` in the section on running models for details on how to use the parallel settings to generate and execute parallel runs.
