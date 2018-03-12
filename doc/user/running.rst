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

The included command-line tool ``calliope run`` will execute a given run configuration::

   $ calliope run my_model/run.yaml

It will generate and solve the model, then save the results to the the output directory given by

``output.path`` in the run configuration.

Saving results
--------------

If running single runs via the command-line tool or using the parallel run functionality, results will be saved as either a single NetCDF file per model run or a set of CSV files per model run. These can then be read back into an interactive Python session for analysis or further processed with any other tool available to the modeller.

Two output formats are available: a collection CSV files or a single NetCDF file. They can be chosen by settings ``output.format`` in the run configuration (set to ``netcdf`` or ``csv``). The :mod:`~calliope.open_netcdf` module provides methods to read results stored in the NetCDF format, so that they can then be analyzed with the :mod:`~calliope.analysis` module.

Overrides
---------

In the command line interface we use ``--override_file=overrides.yaml:milp``. Multiple overrides from the YAML file could be applied at once. E.g. if we changed some costs and had an additional entry ``cost_changes``, we could call ``--override_file=overrides.yaml:milp,cost_changes`` to apply both overrides.

.. seealso::

    :doc:`analysing`, :ref:`building_overrides`

---------------------------------
Running interactively with Python
---------------------------------

An example which also demonstrates some of the analysis possibilities after running a model is given in the following Jupyter notebook, based on the national-scale example model. Note that you can download and run this notebook on your own machine (if both Calliope and the Jupyter Notebook are installed):

:nbviewer_docs:`Calliope interactive national-scale example notebook <_static/notebooks/tutorial.ipynb>`


The most basic way to run a model programmatically from within a Python interpreter is to create a :class:`~calliope.Model` instance with a given ``model.yaml`` configuration file, and then call its :meth:`~calliope.Model.run` method::

   import calliope
   model = calliope.Model(config='/path/to/model_configuration.yaml')
   model.run()

.. note:
    If ``config`` is not specified (i.e. ``model = Model()``), an error is raised. See :doc:`ref_example_models` for information on instantiating a simple example model without specifying a run configuration.

Other ways to load a model interactively are:

    * giving ``config`` an :class:`~calliope.AttrDict` object or standard dictionary, which has the same nested format as the YAML files (top-level keys: ``model``, ``run``, ``locations``, ``techs``)
    * providing a ``model_data`` object, which is the complete pre-processed NetCDF of a model (loaded in as a xarray Dataset).

After instantiating the ``Model`` object, and before calling the ``run()`` method, it is possible to manually inspect and adjust the configuration of the model. The preprocessed inputs are all held in the xarray Dataset ``model.inputs``.

After the model has been solved, an xarray Dataset containing results (``model.results``) can be viewed. At this point the model can be saved :meth:`~calliope.Model.to_csv` and :meth:`~calliope.Model.to_netcdf`, which saves all inputs and results.

Overrides
---------

Interactively we apply this override by setting the `override_file` argument to e.g. ``overrides.yaml:milp`` and/or the `override_dict` argument to a dictionary of overrides.

.. _generating_scripts:

--------------------------------------
Generating scripts for many model runs
--------------------------------------

Scripts to simplify the creation and execution of a large number of Calliope model runs are generated with the ``calliope generate`` command-line tool. More detail on this is available in :ref:`run_config_generate`.

------------------------
Improving solution times
------------------------

Large models will take time to solve. The most basic advice is to just let it run on a remote device (another computer or a high performance computing cluster) and forget about it until it is done. However, if you need results *now*, there are ways to improve solution time, invariably at the expense of model 'accuracy'.

Number of variables
-------------------

The sets ``locs``, ``techs``, ``timesteps``, ``carriers``, and ``costs`` all contribute to model complexity. A reduction of any of these sets will reduce the number of resulting decision variables in the optimisation, which in turn will improve solution times.

.. note::
    By reducing the number of locations (e.g. merging nearby locations) you also remove the technologies linking those locations to the rest of the system, which is additionally beneficial.

Currently, we only provide automatic set reduction for timesteps. Timesteps can be resampled (e.g. 1hr -> 2hr intervals), masked (e.g. 1hr -> 12hr intervals except one week of particular interest), or clustered (e.g. 365 days to 5 days, each representing 73 days of the year, with 1hr resolution). In so doing, significant solution time improvements can be acheived.

.. seealso::
    :ref:`time_clustering`, `Stefan Pfenninger (2017). Dealing with multiple decades of hourly wind and PV time series in energy models: a comparison of methods to reduce time resolution and the planning implications of inter-annual variability. Applied Energy. <https://doi.org/10.1016/j.apenergy.2017.03.051>`_


Complex technologies
--------------------

Calliope is primarily an LP framework, but application of certain constraints will trigger binary or integer decision variables. When triggered, a MILP model will be created.

In both cases, there will be a time penalty, as linear programming solvers are less able to converge on solutions of problems which include binary or integer decision variables. But, the additional functionality can be useful. A purchasing cost allows for a cost curve of the form ``y = Mx + C`` to be applied to a technology, instead of the LP costs which are all of the form ``y = Mx``. Integer units also trigger per-timestep decision variables, which allow technologies to be "on" or "off" at each timestep.

Additionally, in LP models, interactions between timesteps (in ``storage`` technologies) can lead to longer solution time. The exact extent of this is as-yet untested.

TODO: move elsewhere:

By applying a ``purchase`` cost to a technology, that technology will have a binary variable associated with it, describing whether or not it has been "purchased".

By applying ``units.max``, ``units.min``, or ``units.equals`` to a technology, that technology will have a integer variable associated with it, describing how many of that technology have been "purchased". If a ``purchase`` cost has been applied to this same technology, the purchasing cost will be applied per unit.

.. Warning::

   Integer and Binary variables are still experimental and may not cover all edge cases as intended. Please `raise an issue on GitHub <https://github.com/calliope-project/calliope/issues>`_ if you see unexpected behavior.

.. seealso:: :ref:`milp_example_model`

Model mode
----------
Solution time increases more than linearly with the number of decision variables. As it splits the model into ~daily chunks, operational mode can help to aleviate solution time of big problems. This is clearly at the expense of fixing technology capacities. However, one solution is to use a heavily time clustered ``plan`` mode to get indicative model capapcities. Then run ``operate`` mode with these capacities to get a higher resolution operation strategy. If necessary, this process could be iterated.

.. seealso:: :ref:`operational_mode`

Solver choice
-------------
The open-source solvers (``GLPK`` and ``CBC``) are slower than the commercial solvers. If you are an academic researcher, it's recommended to acquire a free licence for ``Gurobi`` or ``CPLEX`` to very quickly improve solution times. Particularly, GPLK suffers in solver MILP models. CBC is an improvement on it, but can be several orders of magnitude slower at reaching a solution than gurobi or CPLEX.

.. seealso:: :ref:`solver_options`

----------------------
Debugging failing runs
----------------------

What will typically go wrong, in order of decreasing likelihood:

   * The model is improperly defined or missing data. Calliope will attempt to diagnose some common errors and raise an appropriate error message.
   * The model is consistent and properly defined but infeasible. Calliope will be able to construct the model and pass it on to the solver, but the solver (after a potentially long time) will abort with a message stating that the model is infeasible.
   * There is a bug in Calliope causing the model to crash either before being passed to the solver, or after the solver has completed and when results are passed back to Calliope.

Calliope provides some run configuration options to make it easier to determine the cause of the first two of these possibilities. See the :ref:`debugging options described in the full configuration listing <debugging_runs_config>`.
