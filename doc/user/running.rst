
=================
Running the model
=================

There are two basic modes for the model: planning mode and operational mode. The mode is set in the run configuration.

In planning mode, constraints are given as upper and lower boundaries and the model decides on an optimal system configuration.

In operational mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm (see :ref:`config_reference_model_wide` for the settings that control the receding horizon).

In either case, there are three ways to run the model:

1. With the ``calliope run`` command-line tool.

2. By generating and then executing parallel runs with the ``calliope generate`` command-line tool.

3. By programmatically creating and running a model from within other Python code.

.. Note:: The solver is set in the run configuration and passed to Pyomo, so any solver that is installed and available for use in Pyomo can be selected.

--------------------------------------
Single runs with the command-line tool
--------------------------------------

The included command-line tool ``calliope run`` will run a given run configuration::

   calliope run my_model/run.yaml

It will generate and solve the model, then save the results to the the output directory given by ``output.path`` in the run configuration.

Two output formats are available: CSV files and HDF, and they can be chosen by settings ``output.format`` in the run configuration (set to ``hdf`` or ``csv``). HDF results in a single compressed file in the high-performance HDF5 data format. The :mod:`~calliope.analysis` module provides methods to read and analyze these HDF files.

For easier analysis via third-party tools, the CSV option saves a set of CSV files into the given output directory.

.. _parallel_runs:

-------------
Parallel runs
-------------

.. Warning:: This functionality is currently not Windows compatible.

Parallel runs are created with the ``calliope generate`` command-line tool as follows:

* Create a ``run.yaml`` file with a ``parallel:`` section as needed (see :ref:`run_config_parallel_runs`).
* On the command line, run ``calliope generate path/to/run.yaml``.
* By default, this will create a new subdirectory inside a ``runs`` directory in the current working directory. You can optionally specify a different target directory by giving another path, e.g. ``calliope generate path/to/run.yaml path/to/my_run_files``.
* Calliope generates several files and directories in the target path. The most important are the ``Runs`` subdirectory which hosts the self-contained configuration for the runs, the ``run.sh`` script, which is responsible for running each run, and the submit.sh script (or scripts), which contain the job control data for a cluster.

The ``run.sh`` script can simply be called with an integer argument from the sequence (1, number of parallel runs) to execute a given run, e.g. ``run.sh 1``, ``run.sh 2``, etc. This way the runs can be executed sequentially on a single machine.

To submit the resulting runs on a cluster with bsub, use ``bsub < submit.sh``, and on a cluster with qsub, ``qsub submit.sh``.

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

``config_run`` can also take an :class:`~calliope.utils.AttrDict` object containing the configuration. Furthermore, ``Model()`` also has an ``override`` argument which takes an ``AttrDict`` with settings that will override the given run settings.

After instantiating the ``Model`` object, and before calling the ``run()`` method, it is possible to manually inspect and adjust the configuration of the model.

If used in an interactive IPython session the model can be queried after running it, e.g.::

   # Get a pandas DataFrame of system variables
   system_vars = model.get_system_variables()

The :doc:`API documentation <../api/api>`, as well as comments in the source code, give an overview of the available methods for programmatic access.
