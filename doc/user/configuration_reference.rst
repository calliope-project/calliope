
=======================
Configuration reference
=======================

.. _yaml_format:

YAML configuration file format
------------------------------

All configuration files (with the exception of time series data files) are in the YAML format, "a human friendly data serialization standard for all programming languages".

Configuration for Calliope is usually specified as ``option: value`` entries, where ``value`` might be a number, a text string, or a list (e.g. a list of further settings).

Calliope allows an abbreviated form for long, nested settings:

.. code-block:: yaml

   one:
      two:
         three: x

can be written as:

.. code-block:: yaml

   one.two.three: x

Calliope also allows a special ``import:`` directive in any YAML file. This can specify one or several YAML files to import. If both the imported file and the current file define the same option, the definition in the current file takes precedence.

Using quotation marks (``'`` or ``"``) to enclose strings is optional, but can help with readability. The three ways of setting ``option`` to ``text`` below are equivalent:

.. code-block:: yaml

   option: "text"
   option: 'text'
   option: text

Sometimes, a setting can be either enabled or disabled, in this case, the boolean values ``true`` or ``false`` are used.

Comments can be inserted anywhere in YAML files with the ``#`` symbol. The remainder of a line after ``#`` is interpreted as a comment.

See the `YAML website <http://www.yaml.org/>`_ for more general information about YAML.

Calliope internally represents the configuration as :class:`~calliope.utils.AttrDict`\ s, which are a subclass of the built-in Python dictionary data type (``dict``) with added functionality such as YAML reading/writing and attribute access to keys.

.. _config_reference_model_wide:

Model-wide settings
-------------------

These settings can either be in a central ``model.yaml`` file, or imported from other files if desired.

Mandatory model-wide settings with no default values (see :doc:`configuration` for more information on defining ``techs``, ``locations`` and ``links``):

.. code-block:: yaml

   data_path: 'data'  # Path to CSV (time series) data files

   techs:
       # ... technology definitions ...

   locations:
       # ... location definitions ...

   links:
       # ... transmission link definitions ...

Optional model-wide settings with no default values (example settings are shown here):

.. code-block:: yaml

   constraints:  # List of additional constraints
       - constraints.ramping.ramping_rate
       # ... other constraints to load ...

   group_fraction:
       # ... setup for group_fraction constraints (see model formulation section) ...

   metadata:  # Metadata for analysis and plotting
       map_boundary: []
       location_coordinates:
       location_ordering:

.. TODO document metadata settings

Optional model-wide settings that have defaults set by Calliope (default values are shown here):

.. code-block:: yaml

   # Chooses the objective function
   # If not set, defaults to the included cost minimization objective
   objective:  'constraints.objective.objective_cost_minimization'

   startup_time: 12  # Length of startup period (hours)

   opmode:  # Operation mode settings
       horizon: 48  # Optimization period length (hours)
       window: 24  # Operation period length (hours)

   system_margin:  # Per-carrier system margins
       power: 0

.. _config_reference_techs:

Technology
----------

A technology with the identifier ``tech_identifier`` is configured by a YAML block within a ``techs`` block. The following block shows all available options and their defaults (see further below for the constraints, costs, and depreciation definitions):

.. code-block:: yaml

   tech_identifier:
       name:  # A descriptive name, e.g. "Offshore wind"
       parent:  # An abstract base technology, or a previously defined one
       stack_weight: 100  # Weight of this technology in the stack when plotting
       color: false  # HTML color code, or `false` to choose a random color
       source_carrier: false # Carrier to consume, for conversion technologies
       group: false  # Make this a group for purposes of supply diversity analysis
       weight: 1.0 # Cost weighting in objective function
       constraints:
           # ... constraint definitions ...
       costs:
           monetary:
               # ... monetary cost definitions ...
           # ... other cost classes ...
       depreciation:
           # ... depreciation definitions ...

Each technology **must** define a ``parent``, which can either be an abstract base technology such as ``supply``, or any other technology previously defined in the model. The technology inherits all settings from its parent, but overwrites anything it specifies again itself. See :ref:`config_parents_and_groups` for more details on this and on the function of the ``group:`` option.

``stack_weight`` and ``color`` determine how the technology is shown in model outputs. The higher the ``stack_weight``, the lower a technology will be shown in stackplots.

The ``depreciation`` definition is optional and only necessary if defaults need to be overridden. However, at least one constraint (such as ``e_cap_max``) and one cost should usually be defined.

Transmission technologies can additionally specify per-distance constraints and per-distance costs (see :ref:`transmission_links`). Currently, only ``e_loss`` constraints and ``e_cap`` costs are supported:

.. code-block:: yaml

  transmission_tech:
     # per_distance constraints specified per 100 units of distance
     per_distance: 100
     constraints_per_distance:
        e_loss: 0.01  # loss per 100 units of distance
     costs_per_distance:
        monetary:
           e_cap: 10  # cost per 100 units of distance

.. _config_reference_constraints:

Technology constraints
----------------------

The following table lists all available technology constraint settings and their default values. All of these can be set by ``tech_identifier.constraints.constraint_name``, e.g. ``nuclear.constraints.e_cap_max``.

.. csv-table::
   :file: includes/default_constraints.csv
   :header: Setting,Default,Details
   :widths: 10, 5, 30
   :stub-columns: 0

Technology costs
----------------

These are all the available costs, which are set to :math:`0` by default for every defined cost class. Costs are set by ``tech_identifier.costs.cost_class.cost_name``, e.g. ``nuclear.costs.monetary.e_cap``.

.. csv-table::
   :file: includes/default_costs.csv
   :header: Setting,Default,Details
   :widths: 10, 5, 30
   :stub-columns: 0

Technology depreciation
------------------------

These technology depreciation settings apply when calculating levelized costs. The interest rate can be set on a per-cost class basis, and defaults to :math:`0.10` for ``monetary`` and :math:`0` for every other cost class.

.. literalinclude:: includes/default_depreciation.yaml
   :language: yaml

Abstract base technologies
--------------------------

This lists all pre-defined abstract base technologies and the defaults they provide. Note that it is not possible to define a technology with the same identifier as one of the abstract base technologies. In addition to providing default values for some options, which abstract base technology a user-defined technology inherits from determines how Calliope treats the technology internally.

Supply and demand
^^^^^^^^^^^^^^^^^

``supply``:

.. literalinclude:: includes/basetech_supply.yaml
   :language: yaml

``demand``:

.. literalinclude:: includes/basetech_demand.yaml
   :language: yaml

Unmet demand
^^^^^^^^^^^^

``unmet_demand``:

.. literalinclude:: includes/basetech_unmet_demand.yaml
   :language: yaml

``unmet_demand_as_supply_tech``:

.. literalinclude:: includes/basetech_unmet_demand_as_supply_tech.yaml
   :language: yaml

Storage, transmission and conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Warning:: The default values provided for ``r`` and ``e_con`` by ``storage``, ``transmission``, and ``conversion`` should not be overridden.

``storage``:

.. literalinclude:: includes/basetech_storage.yaml
   :language: yaml

``transmission``:

.. literalinclude:: includes/basetech_transmission.yaml
   :language: yaml

``conversion``:

.. literalinclude:: includes/basetech_conversion.yaml
   :language: yaml

.. _config_reference_run:

Run settings
------------

These settings will usually be in a central ``run.yaml`` file, which may import from other files if desired.

Mandatory settings:

* ``model``: Path to the model configuration which is to be used for this run
* ``mode``:  ``plan`` or ``operate``, whether to run the model in planning or operational mode
* ``solver``: Name of the solver to use

Optional settings:

* Output options -- these are only used when the model is run via the ``calliope run`` command-line tool:
   * ``output.path``: Path to an output directory to save results (will be created if it doesn't exist already)
   * ``output.format``:  Format to save results in, either ``hdf`` or ``csv``
* ``parallel``: Settings used to generate parallel runs, see :ref:`run_config_parallel_runs` for the available options
* ``time``: Settings to adjust time resolution, see :ref:`run_time_res` for the available options
* ``override``: Override arbitrary settings from the model configuration. E.g., this could specify ``techs.nuclear.costs.monetary.e_cap: 1000`` to set the ``e_cap`` costs of ``nuclear``, overriding whatever was set in the model configuration
* ``solver_options``: A list of options, which are passed on to the chosen solver, and are therefore solver-dependent (see below)

Optional debug settings:

* ``subset_y``, ``subset_x``, ``subset_t``: specify if only a subset of technologies (y), locations (x), or timesteps (t) should be used for this run. This can be useful for debugging purposes. The timestep subset can be specified as ``[startdate, enddate]``, e.g. ``['2005-01-01', '2005-01-31']``. The subsets are processed before building the model and applying time resolution adjustments, so time resolution functions will only see the reduced set of data.
* ``debug.keepfiles``: Whether to keep temporary files (default ``false``), useful to debug model problems
* ``debug.symbolic_solver_labels``: By default, Pyomo does not generate components with human-readable names, which is faster. To debug models (particularly when using ``debug.keepfiles: true``), this setting should also be set to ``true`` so that the generated model becomes human-readable

Solver options
^^^^^^^^^^^^^^

Gurobi: Refer to the `Gurobi manual <http://www.gurobi.com/resources/documentation>`_, which contains a list of parameters. Simply use the names given in the documentation (e.g. "NumericFocus" to set the numerical focus value).

CPLEX: Refer to the `CPLEX documentation <http://www.ibm.com/support/docview.wss?uid=swg21503602>`_, which contains a list of parameters. Use the "Interactive" parameter names, replacing any spaces with underscores (for example, the memory reduction switch is called "emphasis memory", and thus becomes "emphasis_memory").
