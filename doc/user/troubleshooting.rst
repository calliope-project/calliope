---------------
Troubleshooting
---------------

General strategies
------------------

* **Building a smaller model**: :yaml:`model.subset_time` allows specifying a subset of timesteps to be used. This can be useful for debugging purposes as it can dramatically speed up model solution times. The timestep subset can be specified as :yaml:`[startdate, enddate]`, e.g. :yaml:`['2005-01-01', '2005-01-31']`, or as a single time period, such as :yaml:`2005-01` to select January only. The subsets are processed before building the model and applying time resolution adjustments, so time resolution reduction functions will only see the reduced set of data.

* **Retaining logs and temporary files**: The setting :yaml:`run.save_logs`, disabled by default, sets the directory into which to save logs and temporary files from the backend, to inspect solver logs and solver-generated model files. This also turns on symbolic solver labels in the Pyomo backend, so that all model components in the backend model are named according to the corresponding Calliope model components (by default, Pyomo uses short random names for all generated model components).

*
  **Analysing the optimisation problem without running the model**: If you are comfortable with navigating Pyomo objects, then you can build the Pyomo model backend without running the optimisation problem, using :python:`model.run(build_only=True)`. Pyomo objects are then accessible within :python:`model._backend_model`. For instance, the constraint limiting energy capacity can be viewed by calling :python:`model._backend_model.energy_capacity_constraint.pprint('hi')`. Alternatively, if you are working from the command line or have little experience with Pyomo, you can generate an LP file. The LP file contains the mathematical model formulation of a fully built Calliope model. It is a standard format that can be passed to various solvers. Examining the LP file manually or using additional tools (see below) can help find issues when a model is infeasible or unbounded. To build a model and save it to LP without actually solving it, use:

  .. code-block:: shell

    calliope run my_model.yaml --save_lp=my_saved_model.lp

  or, interactively:

  .. code-block:: python

    model.to_lp('my_saved_model.lp')

Improving solution times
------------------------

One way to improve solution time is to reduce the size of a problem (another way is to address potential numerical issues, which is dealt with further below in :ref:`understanding_infeasibility`).

Number of variables
^^^^^^^^^^^^^^^^^^^

The sets ``locs``, ``techs``, ``timesteps``, ``carriers``, and ``costs`` all contribute to model complexity. A reduction of any of these sets will reduce the number of resulting decision variables in the optimisation, which in turn will improve solution times.

.. note::
    By reducing the number of locations (e.g. merging nearby locations) you also remove the technologies linking those locations to the rest of the system, which is additionally beneficial.

Currently, we only provide automatic set reduction for timesteps. Timesteps can be resampled (e.g. 1hr -> 2hr intervals), masked (e.g. 1hr -> 12hr intervals except one week of particular interest), or clustered (e.g. 365 days to 5 days, each representing 73 days of the year, with 1hr resolution). In so doing, significant solution time improvements can be acheived.

.. seealso::
    :ref:`time_clustering`, `Stefan Pfenninger (2017). Dealing with multiple decades of hourly wind and PV time series in energy models: a comparison of methods to reduce time resolution and the planning implications of inter-annual variability. Applied Energy. <https://doi.org/10.1016/j.apenergy.2017.03.051>`_


Complex technologies
^^^^^^^^^^^^^^^^^^^^

Calliope is primarily an LP framework, but application of certain constraints will trigger binary or integer decision variables. When triggered, a MILP model will be created.

In both cases, there will be a time penalty, as linear programming solvers are less able to converge on solutions of problems which include binary or integer decision variables. But, the additional functionality can be useful. A purchasing cost allows for a cost curve of the form :math:`y = Mx + C` to be applied to a technology, instead of the LP costs which are all of the form :math:`y = Mx`. Integer units also trigger per-timestep decision variables, which allow technologies to be "on" or "off" at each timestep.

Additionally, in LP models, interactions between timesteps (in ``storage`` technologies) can lead to longer solution time. The exact extent of this is as-yet untested.

Model mode
^^^^^^^^^^

Solution time increases more than linearly with the number of decision variables. As it splits the model into ~daily chunks, operational mode can help to alleviate solution time of big problems. This is clearly at the expense of fixing technology capacities. However, one solution is to use a heavily time clustered ``plan`` mode to get indicative model capacities. Then run ``operate`` mode with these capacities to get a higher resolution operation strategy. If necessary, this process could be iterated.

.. seealso:: :ref:`operational_mode`

Influence of solver choice on speed
-----------------------------------

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


.. _understanding_infeasibility:

Understanding infeasibility and numerical instability
-----------------------------------------------------

.. note:: A good first step when faced with an infeasible model is often to remove constraints, in particular more complex constraints. For example, different combinations of group constraints can easily introduce mutually exclusive requirements on capacities or output from specific technologies. Once a minimal model works, more complex constraints can be turned on again one after the other.

Using the Gurobi solver
^^^^^^^^^^^^^^^^^^^^^^^

To understand infeasible models:

* Set :yaml:`run.solver_options.DualReductions: 0` to see whether a model is infeasible or unbounded.
* To analyse infeasible models, save an LP file with the :sh:`--save_lp` command-line option, then use Gurobi to generate an Irreducible Inconsistent Subsystem that shows which constraints are infeasible:

  .. code-block:: shell

    gurobi_cl ResultFile=result.ilp my_saved_model.lp

  More detail on this is in the `official Gurobi documentation <https://www.gurobi.com/documentation/current/refman/solving_a_model2.html>`_.

To deal with numerically unstable models, try setting :yaml:`run.solver_options.Presolve: 0`, as large numeric ranges can cause the pre-solver to generate an `infeasible or numerically unstable model <http://www.gurobi.com/documentation/current/refman/numerics_why_scaling_and_g.html>`_. The `Gurobi Guidelines for Numerical Issues <https://www.gurobi.com/documentation/current/refman/numerics_gurobi_guidelines.html>`_ give detailed guidance for strategies to address numerically difficult optimisation problems.

Using the CPLEX solver
^^^^^^^^^^^^^^^^^^^^^^

There are two ways to understand infeasibility when using the CPLEX solver, the first is quick and the second is more involved:

1. Save solver logs for your model (:yaml:`run.save_logs: path/to/log_directory`). In the directory, open the file ending in '.cplex.log' to see the CPLEX solver report. If the model is infeasible or unbounded, the offending constraint will be identified (e.g. "`SOLVER: Infeasible variable = slack c_u_carrier_production_max_constraint(region1_2__csp__power_2005_01_01_07_00_00)_`"). This may be enough to understand why the model is failing, if not...

2. Open the LP file in CPLEX interactive (run `cplex` in the command line to invoke a CPLEX interactive session). The LP file for the problem ends with '.lp' in the log folder (`read path/to/file.lp`). Once loaded, you can try relaxing variables / constraints to see if the problem can be solved with relaxation (`FeasOpt`). You can also identify conflicting constraints (`tools conflict`) and print those constraints directly (`display conflict all`). There are many more commands available to analyse individual constraints and variables in the `Official CPLEX documentation <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/UsrMan/topics/infeas_unbd/partInfeasUnbnded_title_synopsis.html>`_.

Similar to Gurobi, numerically unstable models may lead to unexpected infeasibility, so you can try :yaml:`run.solver_options.preprocessing_presolve: 0` or you can request CPLEX to more aggressively scale the problem itself using the `solver option <https://www.ibm.com/support/knowledgecenter/en/SS9UKU_12.4.0/com.ibm.cplex.zos.help/Parameters/topics/ScaInd.html>`_ ``read_scale: 1`` . The `CPLEX documentation page on numeric difficulties <https://www.ibm.com/support/knowledgecenter/en/SS9UKU_12.4.0/com.ibm.cplex.zos.help/UsrMan/topics/cont_optim/simplex/20_num_difficulty.html>`_ goes into more detail on numeric instability.


Rerunning a model
-----------------

After running, if there is an infeasibility you want to address, or simply a few values you dont think were quite right, you can change them and rerun your model. If you change them in `model.inputs`, just rerun the model as :python:`model.run(force_rerun=True)`.

.. note:: :python:`model.run(force_rerun=True)` will replace you current model.results and rebuild he entire model backend. You may want to save your model before doing this.

Particularly if your problem is large, you may not want to rebuild the backend to change a few small values. Instead you can interface directly with the backend using the :python:`model.backend` functions, to update individual parameter values and switch constraints on/off. By rerunning the backend specifically, you can optimise your problem with these backend changes, without rebuilding the backend entirely.

.. note:: :python:`model.inputs` and :python:`model.results` will not be changed when updating and rerunning the backend. Instead, a new xarray Dataset is returned.

.. seealso:: :ref:`backend_interface`


Debugging model errors
----------------------

Calliope provides a method to save its fully built and commented internal representation of a model to a single YAML file with :python:`Model.save_commented_model_yaml(path)`. Comments in the resulting YAML file indicate where original values were overridden.

Because this is Calliope's internal representation of a model directly before the ``model_data`` ``xarray.Dataset`` is built, it can be useful for debugging possible issues in the model formulation, for example, undesired constraints that exist at specific locations because they were specified model-wide without having been superseded by location-specific settings.

Further processing of the data does occur before solving the model. The final values of parameters used by the backend solver to generate constraints can be analysed when running an interactive Python session by running :python:`model.backend.get_input_params()`. This provides a user with an xarray Dataset which will look very similar to :python:`model.inputs`, except that assumed :ref:`default values <defaults>` will be included. An attempt at running the model has to be made in order to be able to run this command.

.. seealso::

    If using Calliope interactively in a Python session, we recommend reading up on the `Python debugger <https://docs.python.org/3/library/pdb.html>`_ and (if using Jupyter notebooks) making use of the `%debug magic <https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug>`_.
