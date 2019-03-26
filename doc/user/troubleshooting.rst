---------------
Troubleshooting
---------------

General strategies
------------------

* **Building a smaller model**: ``model.subset_time`` allows specifying a subset of timesteps to be used. This can be useful for debugging purposes as it can dramatically speed up model solution times. The timestep subset can be specified as ``[startdate, enddate]``, e.g. ``['2005-01-01', '2005-01-31']``, or as a single time period, such as ``2005-01`` to select January only. The subsets are processed before building the model and applying time resolution adjustments, so time resolution reduction functions will only see the reduced set of data.

* **Retaining logs and temporary files**: The setting ``run.save_logs``, disabled by default, sets the directory into which to save logs and temporary files from the backend, to inspect solver logs and solver-generated model files. This also turns on symbolic solver labels in the Pyomo backend, so that all model components in the backend model are named according to the corresponding Calliope model components (by default, Pyomo uses short random names for all generated model components).

* **Saving an LP file without running the model**: The LP file contains the mathematical model formulation of a fully built Calliope model. It is a standard format that can be passed to various solvers. Examining the LP file manually or using additional tools (see below) can help find issues when a model is infeasible or unbounded. To build a model and save it to LP without actually solving it, use:

  .. code-block:: shell

    calliope run my_model.yaml --save_lp=my_saved_model.lp

Understanding infeasibility and numerical instability
-----------------------------------------------------

Using the Gurobi solver
^^^^^^^^^^^^^^^^^^^^^^^

To understand infeasible models:

* Set ``run.solver_options.DualReductions: 0`` to see whether a model is infeasible or unbounded.
* To analyse infeasible models, save an LP file with the ``--save_lp`` command-line option, then use Gurobi to generate an Irreducible Inconsistent Subsystem that shows which constraints are infeasible:

  .. code-block:: shell

    gurobi_cl ResultFile=result.ilp my_saved_model.lp

  More detail on this is in the `official Gurobi documentation <https://www.gurobi.com/documentation/current/refman/solving_a_model2.html>`_.

To deal with numerically unstable models, try setting ``run.solver_options.Presolve: 0``, as large numeric ranges can cause the pre-solver to generate an infeasible or numerically unstable model. The `Gurobi Guidelines for Numerical Issues <https://www.gurobi.com/documentation/current/refman/numerics_gurobi_guidelines.html>`_ give detailed guidance for strategies to address numerically difficult optimisation problems.

Using the CPLEX solver
^^^^^^^^^^^^^^^^^^^^^^

There are two ways to understand infeasibility when using the CPLEX solver, the first is quick and the second is more involved:

1. Save solver logs for your model (``run.save_logs: path/to/log_directory``). In the directory, open the file ending in '.cplex.log' to see the CPLEX solver report. If the model is infeasible or unbounded, the offending constraint will be identified (e.g. ``SOLVER: Infeasible variable = slack c_u_carrier_production_max_constraint(region1_2__csp__power_2005_01_01_07_00_00)_``). This may be enough to understand why the model is failing, if not...

2. Open the LP file in CPLEX interactive (run `cplex` in the command line to invoke a CPLEX interactive session). The LP file for the problem ends with '.lp' in the log folder (`read path/to/file.lp`). Once loaded, you can try relaxing variables / constraints to see if the problem can be solved with relaxation (`FeasOpt`). You can also identify conflicting constraints (`tools conflict`) and print those constraints directly (`display conflict all`). There are many more commands available to analyse individual constraints and variables in the `Official CPLEX documentation <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/UsrMan/topics/infeas_unbd/partInfeasUnbnded_title_synopsis.html>`_.

Similar to Gurobi, numerically unstable models may lead to unexpected infeasibility, so you can try ``run.solver_options.preprocessing_presolve: 0``. The `CPLEX documentation page on numeric difficulties <https://www.ibm.com/support/knowledgecenter/en/SS9UKU_12.4.0/com.ibm.cplex.zos.help/UsrMan/topics/cont_optim/simplex/20_num_difficulty.html>`_ goes into more detail on numeric instability.

Debugging model errors
----------------------

Calliope provides a method to save its fully built and commented internal representation of a model to a single YAML file with ``Model.save_commented_model_yaml(path)``. Comments in the resulting YAML file indicate where original values were overridden.

Because this is Calliope's internal representation of a model directly before the ``model_data`` ``xarray.Dataset`` is built, it can be useful for debugging possible issues in the model formulation, for example, undesired constraints that exist at specific locations because they were specified model-wide without having been superseded by location-specific settings.

Further processing of the data does occur before solving the model. The final values of parameters used by the backend solver to generate constraints can be analysed when running an interactive Python session by running  ``model.backend.get_input_params()``. This provides a user with an xarray Dataset which will look very similar to ``model.inputs``, except that assumed :ref:`default values <defaults>` will be included. An attempt at running the model has to be made in order to be able to run this command.

.. seealso::

    If using Calliope interactively in a Python session, we recommend reading up on the `Python debugger <https://docs.python.org/3/library/pdb.html>`_ and (if using Jupyter notebooks) making use of the `%debug magic <https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug>`_.
