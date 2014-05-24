
=================
Running the model
=================

There are two basic modes for the model: planning mode and operational mode. The mode is set in ``run.yaml``.

In planning mode, constraints are given as upper and lower boundaries and the model decides on an optimal system configuration.

In operational mode, all capacity constraints are fixed and the system is operated with a receding horizon control algorithm.

------------------
Selecting a solver
------------------

The solver is set in ``run.yaml`` and passed to Pyomo, so any solver that is installed and available for use in Pyomo can be selected. Calliope has been tested with CPLEX (``solver: cplex``) and GLPK (``solver: glpk``).

-----------
Single runs
-----------

The basic way to run a model is to create a ``Model`` instance with a given ``run.yaml`` configuration file, and then call its ``run()`` method:

   import calliope
   model = calliope.Model(config_run='/path/to/run.yaml')
   model.run()

If ``config_run`` is not specified, the built-in example model is used (:ref:`see below <builtin_example>`)

If ``output.save`` has been set to ``true`` in ``run.yaml``, outputs will be saved as CSV files to ``output.path``. If used in an interactive IPython session the model can be queried after running it, e.g.:

   # Returns a pandas DataFrame
   system_vars = model.get_system_variables()
   # Plot system-level variables with matplotlib
   system_vars.plot(figsize=(16, 4))

.. _parallel_runs:

-------------
Parallel runs
-------------

Parallel runs are created with the ``calliope_run.py`` command-line tool as follows:

* Create a ``run.yaml`` file with a ``parallel:`` section as needed.
* On the command line, run ``calliope_run.py path/to/your/run.yaml``.
* By default, this will create a new subdirectory inside a ``runs`` directory in the current working directory. You can specify the target directory with the ``-d/--dir`` command-line option.
* The output directory contains a script that can either be run directly or submitted to a cluster controller (e.g. ``qsub run.sh`` or ``bsub run.sh``)

.. _builtin_example:

--------------------------
The built-in example model
--------------------------

If creating a ``Model()`` without any additional arguments, the built-in example model configuration is used.

..TODO more detail

For more detail on configuring a model either based on the example or entirely from scratch, refer to :doc:`configuration` and :doc:`data`.
