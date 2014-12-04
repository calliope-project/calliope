
=================
Run configuration
=================

The run settings take care of setting up and running the model with a given model configuration. It can be operated in two modes:

1. Directly instantiate an instance of :class:`~calliope.Model` and run it by calling its ``run()`` method. This ignores any settings in the ``parallel`` block in the run settings.
2. Set up a series of parallel runs via the ``calliope_run.py`` command-line tool (:ref:`explained here <parallel_runs>`). This will result in a set of scripts to perform the desired model runs either locally or on a remote cluster.

In case (1), the run settings file can be specified by setting ``config_run`` when instantiating the Model, e.g. ``calliope.Model(config_run='/path/to/run.yaml')``. In case (2), the run settings file is a required argument to ``calliope_run.py``.


.. _run_time_res:

--------------------------
Time resolution adjustment
--------------------------

The default time step length is 1 hour. However, this 1-hourly resolution can be adjusted over parts of the dataset by using the :class:`~calliope.TimeSummarizer` class (currently, only support for downsampling is implemented).

There are two ways to adjust resolution:

1. The :meth:`calliope.TimeSummarizer.reduce_resolution` method: reduces resolution over the entire range of data to the given resolution.
2. The :meth:`calliope.TimeSummarizer.dynamic_timestepper` method: reduces resolution dynamically according to a given mask, allowing to keep high resolution in areas of interest while reducing computational complexity elsewhere.

Dynamic timesteps and masks
---------------------------

In order to use ``dynamic_timestepper``, a mask needs to be generated first.

A mask is a pandas DataFrame with the same index as the data it applies to, and a column called ``summarize`` (in addition to any number of additional columns, which are ignored). The ``summarize`` column containts ``0`` for timesteps that aren't touched, and blocks starting with an integer >1 and followed by the integer's value number of ``-1``, for timesteps that are to be summarized. For example::

   [0, 0, 0, 3, -1, -1, 3, -1, -1, 0, 0, 0]

The above example means "compress the 4th-6th and 7th-9th timesteps into two new timesteps with a resolution of 3".

Functions to generate masks and resolution series are in ``calliope.time_functions``.

.. FIXME this needs updating

A fully-functioning example of using a mask to collapse periods where solar irradiance is zero (i.e., the night) into single timesteps::

   model = calliope.Model()
   s = calliope.TimeSummarizer()
   mask = calliope.time_functions.mask_where_zero(model.data, tech='csp', var='r')
   s.dynamic_timestepper(model.data, mask)  # Modifies data in place
   model.run()

--------------------------
Settings for parallel runs
--------------------------

The run settings can (but do not have to) define a ``parallel:`` section. This section is parsed when using the ``calliope_run.py`` command-line tool to generate a set of runs to be run in parallel (:ref:`explained here <parallel_runs>`).

The available options are detailed in the example model's run settings (``run.yaml``).
