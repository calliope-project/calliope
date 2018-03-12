===============
Running a model
===============

There are essentially three ways to run a Calliope model:

1. With the ``calliope run`` command-line tool.

2. By programmatically creating and running a model from within other Python code, or in an interactive Python session.

3. By generating and then executing scripts with the ``calliope generate_runs`` command-line tool, which is primarily designed for running many scenarios on a high-performance cluster.

----------------------------------
Running with the command-line tool
----------------------------------

We can now run this model::

   $ calliope run testmodel/model.yaml

Because of the output options set in ``model.yaml``, model results will be stored as a set of CSV files in the directory ``Output``. Saving CSV files is an easy way to get results in a format suitable for further processing with other tools. In order to make use of Calliope's analysis functionality, results should be saved as a single NetCDF file instead, which comes with improved performance and handling.

See :doc:`running` for more on how to run a model and then retrieve results from it. See :doc:`analysis` for more details on analyzing results, including the built-in functionality to read results from either CSV or NetCDF files, making them available for further analysis as described above (:ref:`tutorial_run_interactively`).


The included command-line tool ``calliope run`` will execute a given run configuration::

   $ calliope run my_model/run.yaml

It will generate and solve the model, then save the results to the the output directory given by

``output.path`` in the run configuration.

Saving results
--------------

If running single runs via the command-line tool or using the parallel run functionality, results will be saved as either a single NetCDF file per model run or a set of CSV files per model run. These can then be read back into an interactive Python session for analysis -- see :doc:`analysis` -- or further processed with any other tool available to the modeller.

Two output formats are available: a collection CSV files or a single NetCDF file. They can be chosen by settings ``output.format`` in the run configuration (set to ``netcdf`` or ``csv``). The :mod:`~calliope.read` module provides methods to read results stored in either of these formats, so that they can then be analyzed with the :mod:`~calliope.analysis` module.

Overrides
---------

In the command line interface we use ``--override_file=overrides.yaml:milp``. Multiple overrides from the YAML file could be applied at once. E.g. if we changed some costs and had an additional entry ``cost_changes``, we could call ``--override_file=overrides.yaml:milp,cost_changes`` to apply both overrides.


---------------------------------
Running interactively with Python
---------------------------------

An example which also demonstrates some of the analysis possibilities after running a model is given in the following Jupyter notebook, based on the national-scale example model. Note that you can download and run this notebook on your own machine (if both Calliope and the Jupyter Notebook are installed):

:nbviewer_docs:`Calliope interactive national-scale example notebook <_static/notebooks/tutorial.ipynb>`


The most basic way to run a model programmatically from within a Python interpreter is to create a :class:`~calliope.Model` instance with a given ``run.yaml`` configuration file, and then call its :meth:`~calliope.Model.run` method::

   import calliope
   model = calliope.Model(config_run='/path/to/run_configuration.yaml')
   model.run()

If ``config_run`` is not specified (i.e. ``model = Model()``), an error is raised. See :doc:`example_models` for information on instantiating a simple example model without specifying a run configuration.

``config_run`` can also take an :class:`~calliope.utils.AttrDict` object containing the configuration. Furthermore, ``Model()`` has an ``override`` parameter, which takes an ``AttrDict`` with settings that will override the given run settings.

After instantiating the ``Model`` object, and before calling the ``run()`` method, it is possible to manually inspect and adjust the configuration of the model.

After the model has been solved, an xarray Dataset containing solution variables and aggregated statistics is accessible under the ``solution`` property on the model instance.

The :doc:`API documentation <../api/api>` gives an overview of the available methods for programmatic access.

Overrides
---------

Interactively we apply this override by setting the override_file argument to ``overrides.yaml:milp``.

.. _generating_scripts:

--------------------------------------
Generating scripts for many model runs
--------------------------------------

Scripts to simplify the creation and execution of a large number of Calliope model runs are generated with the ``calliope generate`` command-line tool. More detail on this is available in :ref:`run_config_generate`.

------------------------
Improving solution times
------------------------

TBA

Running a Linear (LP) or Mixed Integer Linear (MILP) model
----------------------------------------------------------

Calliope is primarily an LP framework, but application of certain constraints will trigger binary or integer decision variables. When triggered, a MILP model will be created.

By applying a ``purchase`` cost to a technology, that technology will have a binary variable associated with it, describing whether or not it has been "purchased".

By applying ``units.max``, ``units.min``, or ``units.equals`` to a technology, that technology will have a integer variable associated with it, describing how many of that technology have been "purchased". If a ``purchase`` cost has been applied to this same technology, the purchasing cost will be applied per unit.

In both cases, there will be a time penalty, as linear programming solvers are less able to converge on solutions of problems which include binary or integer decision variables. But, the additional functionality can be useful. A purchasing cost allows for a cost curve of the form ``y = Mx + C`` to be applied to a technology, instead of the LP costs which are all of the form ``y = Mx``. Integer units also trigger per-timestep decision variables, which allow technologies to be "on" or "off" at each timestep.

.. Warning::

   Integer and Binary variables are still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

----------------------
Debugging failing runs
----------------------

What will typically go wrong, in order of decreasing likelihood:

   * The model is improperly defined or missing data. Calliope will attempt to diagnose some common errors and raise an appropriate error message.
   * The model is consistent and properly defined but infeasible. Calliope will be able to construct the model and pass it on to the solver, but the solver (after a potentially long time) will abort with a message stating that the model is infeasible.
   * There is a bug in Calliope causing the model to crash either before being passed to the solver, or after the solver has completed and when results are passed back to Calliope.

Calliope provides some run configuration options to make it easier to determine the cause of the first two of these possibilities. See the :ref:`debugging options described in the full configuration listing <debugging_runs_config>`.
