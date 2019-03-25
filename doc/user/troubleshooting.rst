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

To deal with numerically unstable models:

* Try setting ``run.solver_options.Presolve: 0``, as large numeric ranges can cause the pre-solver to generate an infeasible or numerically unstable model.

.. seealso::

    The `Gurobi Guidelines for Numerical Issues <https://www.gurobi.com/documentation/current/refman/numerics_gurobi_guidelines.html>`_ give detailed guidance for strategies to address numerically difficult optimisation problems.

Debugging model errors
----------------------

Calliope provides a method to save its fully built and commented internal representation of a model to a single YAML file with ``Model.save_commented_model_yaml(path)``. Comments in the resulting YAML file indicate where original values were overridden.

Because this is Calliope's internal representation of a model directly before the ``model_data`` ``xarray.Dataset`` is built, it can be useful for debugging possible issues in the model formulation, for example, undesired constraints that exist at specific locations because they were specified model-wide without having been superseded by location-specific settings.

.. seealso::

    If using Calliope interactively in a Python session, we recommend reading up on the `Python debugger <https://docs.python.org/3/library/pdb.html>`_ and (if using Jupyter notebooks) making use of the `%debug magic <https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug>`_.
