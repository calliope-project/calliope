===============
Running a model
===============

There are essentially three ways to run a Calliope model:

1. With the ``calliope run`` command-line tool.

2. By programmatically creating and running a model from within other Python code, or in an interactive Python session.

3. By generating and then executing scripts with the ``calliope generate_runs`` command-line tool, which is primarily designed for running many scenarios on a high-performance cluster.

.. _running_cli:

----------------------------------
Running with the command-line tool
----------------------------------

We can easily run a model after creating it (see :doc:`building`), saving results to a single NetCDF file for further processing::

   $ calliope run testmodel/model.yaml --save_netcdf=results.nc

The ``calliope run`` command takes the following options:

* ``--save_netcdf={filename.nc}``: Save complete model, including results, to the given NetCDF file. This is the recommended way to save model input and output data into a single file, as it preserves all data fully, and allows later reconstruction of the Calliope model for further analysis.
* ``--save_csv={directory name}``: Save results as a set of CSV files to the given directory. This can be handy if the modeler needs results in a simple text-based format for further processing with a tool like Microsoft Excel.
* ``--save_plots={filename.html}``: Save interactive plots to the given HTML file (see :doc:`analysing` for further details on the plotting functionality).
* ``--debug``: Run in debug mode, which prints more internal information, and is useful when troubleshooting failing models.
* ``--scenario={scenario}`` and ``--override_dict={yaml_string}``: Specify a scenario, or one or several overrides, to apply to the model, or apply specific overrides from a YAML string (see below for more information)
* ``--help``: Show all available options.

Multiple options can be specified, for example, saving NetCDF, CSV, and HTML plots simultaneously::

   $ calliope run testmodel/model.yaml --save_netcdf=results.nc --save_csv=outputs --save_plots=plots.html

.. Warning:: Unlike in versions prior to 0.6.0, the command-line tool in Calliope 0.6.0 and upward does not save results by default -- the modeller must specify one of the ``-save`` options.

.. _applying_scenario_or_override:

Applying a scenario or override
-------------------------------

The ``--scenario`` can be used in three different ways:

* It can be given the name of a scenario defined in the model configuration, as in ``--scenario=my_scenario``
* It can be given the name of a single override defined in the model configuration, as in ``--scenario=my_override``
* It can be given a comma-separated string of several overrides defined in the model configuration, as in ``--scenario=my_override_1,my_override_2``

In the latter two cases, the given override(s) is used to implicitly create a "scenario" on-the-fly when running the model. This allows quick experimentation with different overrides without explicitly defining a scenario combining them.

Assuming we have specified an override called ``milp`` in our model configuration, we can apply it to our model with::

   $ calliope run testmodel/model.yaml --scenario=milp --save_netcdf=results.nc

Note that if both a scenario and an override with the same name, such as ``milp`` in the above example, exist, Calliope will raise an error, as it will not be clear which one the user wishes to apply.

It is also possible to use the `--override_dict` option to pass a YAML string that will be applied after anything applied through ``--scenario``::

    $ calliope run testmodel/model.yaml --override_dict="{'model.subset_time': ['2005-01-01', '2005-01-31']}" --save_netcdf=results.nc

.. seealso::

    :doc:`analysing`, :ref:`building_overrides`

---------------------------------
Running interactively with Python
---------------------------------

The most basic way to run a model programmatically from within a Python interpreter is to create a :class:`~calliope.Model` instance with a given ``model.yaml`` configuration file, and then call its :meth:`~calliope.Model.run` method::

   import calliope
   model = calliope.Model('path/to/model.yaml')
   model.run()

.. note:: If ``config`` is not specified (i.e. ``model = Model()``), an error is raised. See :doc:`ref_example_models` for information on instantiating a simple example model without specifying a custom model configuration.

.. note:: Calliope logs useful progress information to the INFO log level, but does not change the default log level from WARNING. To see progress information when running interactively, call ``calliope.set_log_level('INFO')`` immediately after importing Calliope.

Other ways to load a model interactively are:

* Passing an :class:`~calliope.AttrDict` or standard Python dictionary to the :class:`~calliope.Model` constructor, with the same nested format as the YAML model configuration (top-level keys: ``model``, ``run``, ``locations``, ``techs``).
* Loading a previously saved model from a NetCDF file with ``model = calliope.read_netcdf('path/to/saved_model.nc')``. This can either be a pre-processed model saved before its ``run`` method was called, which will include input data only, or a completely solved model, which will include input and result data.

After instantiating the ``Model`` object, and before calling the ``run()`` method, it is possible to manually inspect and adjust the configuration of the model. The pre-processed inputs are all held in the xarray Dataset ``model.inputs``.

After the model has been solved, an xarray Dataset containing results (``model.results``) can be accessed. At this point, the model can be saved with either :meth:`~calliope.Model.to_csv` or :meth:`~calliope.Model.to_netcdf`, which saves all inputs and results, and is equivalent to the corresponding ``--save`` options of the command-line tool.

.. seealso::
    An example of interactive running in a Python session, which also demonstrates some of the analysis possibilities after running a model, is given in the :doc:`tutorials <tutorials>`. You can download and run the embedded notebooks on your own machine (if both Calliope and the Jupyter Notebook are installed).

Scenarios and overrides
-----------------------

There are two ways to override a base model when running interactively, analogously to the use of the command-line tool (see :ref:`applying_scenario_or_override` above):

1. By setting the `scenario` argument, e.g.:

    .. code-block:: python

        model = calliope.Model('model.yaml', scenario='milp')

2. By passing the `override_dict` argument, which is a Python dictionary, an :class:`~calliope.AttrDict`, or a YAML string of overrides:

    .. code-block:: python

        model = calliope.Model(
            'model.yaml',
            override_dict={'run.solver': 'gurobi'}
        )

Tracking progress
-----------------

When running Calliope in command line, logging of model pre-processing and solving occurs automatically. Interactively, for example in a Jupyter notebook, you can enable verbose logging by running the following code before instantiating and running a Calliope model:

.. code-block:: python

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    logger = logging.getLogger()

This will include model processing output, as well as the output of the chosen solver.

--------------------------------------
Generating scripts for many model runs
--------------------------------------

Scripts to simplify the creation and execution of a large number of Calliope model runs are generated with the ``calliope generate_runs`` command-line tool. More detail on this is available in :ref:`generating_scripts`.

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

Model mode
----------

Solution time increases more than linearly with the number of decision variables. As it splits the model into ~daily chunks, operational mode can help to alleviate solution time of big problems. This is clearly at the expense of fixing technology capacities. However, one solution is to use a heavily time clustered ``plan`` mode to get indicative model capacities. Then run ``operate`` mode with these capacities to get a higher resolution operation strategy. If necessary, this process could be iterated.

.. seealso:: :ref:`operational_mode`

Solver choice
-------------

The open-source solvers (GLPK and CBC) are slower than the commercial solvers. If you are an academic researcher, it is recommended to acquire a free licence for Gurobi or CPLEX to very quickly improve solution times. GLPK in particular is slow when solving MILP models. CBC is an improvement, but can still be several orders of magnitude slower at reaching a solution than Gurobi or CPLEX.

We tested solution time for various solver choices on our example models, extended to run over a full year (8760 hours). These runs took place on the University of Cambridge high performance computing cluster, with a maximum run time of 5 hours. As can be seen, CBC is far superior to GLPK. If introducing binary constraints, although CBC is an improvement on GLPK, access to a commercial solver is preferable.

**National scale example model size**

- Variables : 420526 [Nneg: 219026, Free: 105140, Other: 96360]
- Linear constraints : 586972 [Less: 315373, Greater: 10, Equal: 271589]

**MILP urban scale example model**

- Variables: 586996 [Nneg: 332913, Free: 78880, Binary: 2, General Integer: 8761, Other: 166440]
- Linear constraints: 788502 [Less: 394226, Greater: 21, Equal: 394255]

**Solution time**

+-------------------+----------------+
|Solver             |Solution time   |
|                   +--------+-------+
|                   |National|Urban  |
+===================+========+=======+
|GLPK               |4:35:40 |>5hrs  |
+-------------------+--------+-------+
|CBC                |0:04:45 |0:52:13|
+-------------------+--------+-------+
|Gurobi (1 thread)  |0:02:08 |0:03:21|
+-------------------+--------+-------+
|CPLEX (1 thread)   |0:04:55 |0:05:56|
+-------------------+--------+-------+
|Gurobi (4 thread)  |0:02:27 |0:03:08|
+-------------------+--------+-------+
|CPLEX (4 thread)   |0:02:16 |0:03:26|
+-------------------+--------+-------+


.. seealso:: :ref:`solver_options`

Rerunning a model
-----------------

After running, if there is an infeasibility you want to address, or simply a few values you dont think were quite right, you can change them and rerun your model. If you change them in `model.inputs`, just rerun the model as ``model.run(force_rerun=True)``.

.. note:: ``model.run(force_rerun=True)`` will replace you current model.results and rebuild he entire model backend. You may want to save your model before doing this.

Particularly if your problem is large, you may not want to rebuild the backend to change a few small values. Instead you can interface directly with the backend using the ``model.backend`` functions, to update individual parameter values and switch constraints on/off. By rerunning the backend specifically, you can optimise your problem with these backend changes, without rebuilding the backend entirely.

.. note:: ``model.inputs`` and ``model.results`` will not be changed when updating and rerunning the backend. Instead, a new xarray Dataset is returned.

.. seealso:: :ref:`backend_interface`

----------------------
Debugging failing runs
----------------------

What will typically go wrong, in order of decreasing likelihood:

   * The model is improperly defined or missing data. Calliope will attempt to diagnose some common errors and raise an appropriate error message.
   * The model is consistent and properly defined but infeasible. Calliope will be able to construct the model and pass it on to the solver, but the solver (after a potentially long time) will abort with a message stating that the model is infeasible.
   * There is a bug in Calliope causing the model to crash either before being passed to the solver, or after the solver has completed and when results are passed back to Calliope.

Calliope provides help in diagnosing model issues. See the section on :ref:`debugging failing runs <debugging_runs_config>`.
