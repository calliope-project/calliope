
=================
Running the model
=================

There are two basic modes for the model: planning mode and operational mode. The mode is set in the run configuration.

In planning mode, constraints are given as upper and lower boundaries and the model decides on an optimal system configuration.

In operational mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm (see :ref:`config_reference_model_wide` for the settings that control the receding horizon).

In either case, there are three ways to run the model:

1. With the ``calliope run`` command-line tool.

2. By generating and then executing parallel runs with the ``calliope generate`` command-line tool.

3. By programmatically creating and running a model from within other Python code, or in an interactive Python session.

--------------------------------------
Single runs with the command-line tool
--------------------------------------

The included command-line tool ``calliope run`` will execute a given run configuration::

   $ calliope run my_model/run.yaml

It will generate and solve the model, then save the results to the the output directory given by ``output.path`` in the run configuration.

Two output formats are available: a collection CSV files or a single NetCDF file. They can be chosen by settings ``output.format`` in the run configuration (set to ``netcdf`` or ``csv``). The :mod:`~calliope.read` module provides methods to read results stored in either of these formats, so that they can then be analyzed with the :mod:`~calliope.analysis` module.

.. _parallel_runs:

-------------
Parallel runs
-------------

.. Warning:: This functionality is currently not Windows-compatible.

Scripts to simplify the creation and execution of a large number of Calliope model runs are generated with the ``calliope generate`` command-line tool as follows:

* Create a ``run.yaml`` file with a ``parallel:`` section as needed (see :ref:`run_config_parallel_runs`).
* On the command line, run ``calliope generate path/to/run.yaml``.
* By default, this will create a new subdirectory inside a ``runs`` directory in the current working directory. You can optionally specify a different target directory by passing a second path to ``calliope generate``, e.g. ``calliope generate path/to/run.yaml path/to/my_run_files``.
* Calliope generates several files and directories in the target path. The most important are the ``Runs`` subdirectory which hosts the self-contained configuration for the runs and ``run.sh`` script, which is responsible for executing each run. In order to execute these runs in parallel on a compute cluster, a submit.sh script is also generated containing job control data, and which can be submitted via a cluster controller (e.g., ``qsub submit.sh``).

The ``run.sh`` script can simply be called with an integer argument from the sequence (1, number of parallel runs) to execute a given run, e.g. ``run.sh 1``, ``run.sh 2``, etc. This way the runs can easily be executed irrespective of the parallel computing environment available.

.. Note:: Models generated via ``calliope generate`` automatically save results as a single NetCDF file per run inside the parallel runs' ``Output`` subdirectory, regardless of whether the ``output.path`` or ``output.format`` options have been set.

See :ref:`run_config_parallel_runs` for details on configuring parallel runs.

.. _builtin_example:

-----------------------------------------------
Running programmatically from other Python code
-----------------------------------------------

The most basic way to run a model programmatically from within a Python interpreter is to create a :class:`~calliope.Model` instance with a given ``run.yaml`` configuration file, and then call its :meth:`~calliope.Model.run` method::

   import calliope
   model = calliope.Model(config_run='/path/to/run_configuration.yaml')
   model.run()

If ``config_run`` is not specified (e.g. ``model = Model()``), the built-in example model is used (see :doc:`example_model`).

``config_run`` can also take an :class:`~calliope.utils.AttrDict` object containing the configuration. Furthermore, ``Model()`` has an ``override`` parameter, which takes an ``AttrDict`` with settings that will override the given run settings.

After instantiating the ``Model`` object, and before calling the ``run()`` method, it is possible to manually inspect and adjust the configuration of the model.

After the model has been solved, an xarray Dataset containing solution variables and aggregated statistics is accessible under the ``solution`` property on the model instance.

The :doc:`API documentation <../api/api>` gives an overview of the available methods for programmatic access.
