===============
Running a model
===============

There are essentially three ways to run a Calliope model:

1. With the :sh:`calliope run` command-line tool.

2. By programmatically creating and running a model from within other Python code, or in an interactive Python session.

3. By generating and then executing scripts with the :sh:`calliope generate_runs` command-line tool, which is primarily designed for running many scenarios on a high-performance cluster.

.. _running_cli:

----------------------------------
Running with the command-line tool
----------------------------------

We can easily run a model after creating it (see :doc:`building`), saving results to a single NetCDF file for further processing

  .. code-block:: fishshell

    $ calliope run testmodel/model.yaml --save_netcdf=results.nc

The :sh:`calliope run` command takes the following options:

* :sh:`--save_netcdf={filename.nc}`: Save complete model, including results, to the given NetCDF file. This is the recommended way to save model input and output data into a single file, as it preserves all data fully, and allows later reconstruction of the Calliope model for further analysis.
* :sh:`--save_csv={directory name}`: Save results as a set of CSV files to the given directory. This can be handy if the modeler needs results in a simple text-based format for further processing with a tool like Microsoft Excel.
* :sh:`--debug`: Run in debug mode, which prints more internal information, and is useful when troubleshooting failing models.
* :sh:`--scenario={scenario}` and :sh:`--override_dict={yaml_string}`: Specify a scenario, or one or several overrides, to apply to the model, or apply specific overrides from a YAML string (see below for more information)
* :sh:`--help`: Show all available options.

Multiple options can be specified, for example, saving NetCDF, CSV, and HTML plots simultaneously

  .. code-block:: fishshell

   $ calliope run testmodel/model.yaml --save_netcdf=results.nc --save_csv=outputs

.. Warning:: Unlike in versions prior to 0.6.0, the command-line tool in Calliope 0.6.0 and upward does not save results by default -- the modeller must specify one of the :sh:`-save` options.

.. _applying_scenario_or_override:

Applying a scenario or override
-------------------------------

The :sh:`--scenario` can be used in three different ways:

* It can be given the name of a scenario defined in the model configuration, as in :sh:`--scenario=my_scenario`
* It can be given the name of a single override defined in the model configuration, as in :sh:`--scenario=my_override`
* It can be given a comma-separated string of several overrides defined in the model configuration, as in :sh:`--scenario=my_override_1,my_override_2`

In the latter two cases, the given override(s) is used to implicitly create a "scenario" on-the-fly when running the model. This allows quick experimentation with different overrides without explicitly defining a scenario combining them.

Assuming we have specified an override called ``milp`` in our model configuration, we can apply it to our model with

  .. code-block:: fishshell

   $ calliope run testmodel/model.yaml --scenario=milp --save_netcdf=results.nc

Note that if both a scenario and an override with the same name, such as ``milp`` in the above example, exist, Calliope will raise an error, as it will not be clear which one the user wishes to apply.

It is also possible to use the `--override_dict` option to pass a YAML string that will be applied after anything applied through :sh:`--scenario`

  .. code-block:: fishshell

    $ calliope run testmodel/model.yaml --override_dict="{'model.time_subset': ['2005-01-01', '2005-01-31']}" --save_netcdf=results.nc

.. seealso::

    :doc:`analysing`, :ref:`building_overrides`

---------------------------------
Running interactively with Python
---------------------------------

The most basic way to run a model programmatically from within a Python interpreter is to create a :class:`~calliope.Model` instance with a given ``model.yaml`` configuration file, and then call its :meth:`~calliope.Model.run` method::

   import calliope
   model = calliope.Model('path/to/model.yaml')
   model.run()

.. note:: If ``config`` is not specified (i.e. :python:`model = Model()`), an error is raised. See :doc:`ref_example_models` for information on instantiating a simple example model without specifying a custom model configuration.

Other ways to load a model interactively are:

* Passing an :class:`~calliope.AttrDict` or standard Python dictionary to the :class:`~calliope.Model` constructor, with the same nested format as the YAML model configuration (top-level keys: ``model``, ``run``, ``locations``, ``techs``).
* Loading a previously saved model from a NetCDF file with :python:`model = calliope.read_netcdf('path/to/saved_model.nc')`. This can either be a pre-processed model saved before its ``run`` method was called, which will include input data only, or a completely solved model, which will include input and result data.

After instantiating the ``Model`` object, and before calling the ``run()`` method, it is possible to manually inspect and adjust the configuration of the model. The pre-processed inputs are all held in the xarray Dataset :python:`model.inputs`.

After the model has been solved, an xarray Dataset containing results (:python:`model.results`) can be accessed. At this point, the model can be saved with either :meth:`~calliope.Model.to_csv` or :meth:`~calliope.Model.to_netcdf`, which saves all inputs and results, and is equivalent to the corresponding :sh:`--save` options of the command-line tool.

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
            override_dict={'config.solve.solver': 'gurobi'}
        )

.. note:: Both `scenario` and `override_dict` can be defined at once. They will be applied in order, such that scenarios are applied first, followed by dictionary overrides. As such, the `override_dict` can be used to override scenarios.

Tracking progress
-----------------

When running Calliope in the command line, logging of model pre-processing and solving occurs automatically. Interactively, for example in a Jupyter notebook, you can enable verbose logging by setting the log level using :python:`calliope.set_log_verbosity(level)` immediately after importing the Calliope package. By default, :python:`calliope.set_log_verbosity()` also sets the log level for the backend model to `DEBUG`, which turns on output of solver output. This can be disabled by :python:`calliope.set_log_verbosity(level, include_solver_output=False)`. Possible log levels are (from least to most verbose):

1. `CRITICAL`: only show critical errors.
2. `ERROR`: only show errors.
3. `WARNING`: show errors and warnings (default level).
4. `INFO`: show errors, warnings, and informative messages. Calliope uses the INFO level to show a message at each stage of pre-processing, sending the model to the solver, and post-processing, including timestamps.
5. `DEBUG`: SOLVER logging, with heavily verbose logging of a number of function outputs. Only for use when troubleshooting failing runs or developing new functionality in Calliope.

--------------------------------------
Generating scripts for many model runs
--------------------------------------

Scripts to simplify the creation and execution of a large number of Calliope model runs are generated with the :sh:`calliope generate_runs` command-line tool. More detail on this is available in :ref:`generating_scripts`.

------------------------
Improving solution times
------------------------

Large models will take time to solve. The easiest is often to just let a model run on a remote device (another computer, or a high performance computing cluster) and forget about it until it is done. However, if you need results *now*, there are ways to improve solution time.

Details on strategies to improve solution times are given in :doc:`troubleshooting`.

----------------------
Debugging failing runs
----------------------

What will typically go wrong, in order of decreasing likelihood:

   * The model is improperly defined or missing data. Calliope will attempt to diagnose some common errors and raise an appropriate error message.
   * The model is consistent and properly defined but infeasible. Calliope will be able to construct the model and pass it on to the solver, but the solver (after a potentially long time) will abort with a message stating that the model is infeasible.
   * There is a bug in Calliope causing the model to crash either before being passed to the solver, or after the solver has completed and when results are passed back to Calliope.

Calliope provides help in diagnosing all of these model issues. For details, see :doc:`troubleshooting`.
