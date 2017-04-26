
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

.. TODO improve the docs on this warning as well as the underlying messy implementation. perhaps more specific docs on generating parallel runs and the quirks that entails.

.. Warning:: When generating parallel runs with the ``calliope generate`` command-line tool, any ``import`` directive, unlike other settings that point to file system paths such as ``model_override`` or ``data_path``, is evaluated immediately and all imported files are combined into one model configuration file for the parallel runs. This means that while paths used in ``import`` directives don't need adjustment for parallel runs, other settings that work with file system paths probably do need adjustment to account for the way files are laid out on the system running the parallel runs. For this purpose, the ``data_path_adjustment`` inside a ``parallel`` configuration block can change the data path for parallel runs only.

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
       - constraints.optional.ramping_rate
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
       carrier: false # Energy carrier to produce/consume, for all except conversion
       stack_weight: 100  # Weight of this technology in the stack when plotting
       color: false  # HTML color code, or `false` to choose a random color
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

Each technology **must** define a ``parent`, which can either be an abstract base technology such as ``supply``, or any other technology previously defined in the model. The technology inherits all settings from its parent, but overwrites anything it specifies again itself. See :ref:`config_parents_and_groups` for more details on this and on the function of the ``group:`` option.

Each technology **must** also define at least one of the ``carrier`` options. ``carrier`` implicitly defines ``carrier_in`` & ``carrier_out`` for storage and transmission technologies, ``carrier_in`` for demand technologies, and ``carrier_out`` for supply/supply_plus technologies. Supply and demand technologies can be defined using ``carrier_in``/``carrier_out`` instead, which will produce the same result. For conversion and conversion_plus, there are further options available:

.. code-block:: yaml

    tech_identifier:
        primary_carrier: false # Setting the primary carrier_out to associate with costs & constraints, if multiple primary carriers are assigned
        carrier_in: false # Primary energy carrier(s) to consume
        carrier_in_2: false # Secondary energy carrier(s) to consume, conversion_plus only
        carrier_in_3: false # Tertiary energy carrier(s) to consume, conversion_plus only
        carrier_out: false # Primary energy carrier(s) to produce
        carrier_out_2: false # Secondary energy carrier(s) to produce, conversion_plus only
        carrier_out_3: false # Tertiary energy carrier(s) to produce, conversion_plus only

If carriers are given at secondary or tertiary level, they are given in an indented list, with their consumption/production with respect to ``carrier_in``/``carrier_out``. For example:

.. code-block:: yaml

    tech_identifier_1:
        carrier_in: 'primary_consumed_carrier'
        carrier_in_2:
            secondary_consumed_carrier: 0.8 # consumes 0.8 units of ``secondary_consumed_carrier`` for every 1 unit of ``primary_consumed_carrier``
        carrier_in_3:
            tertiary_consumed_carrier: 0.1 # consumes 0.1 units of ``tertiary_consumed_carrier`` for every 1 unit of ``primary_consumed_carrier``
        carrier_out: 'primary_produced_carrier'
        carrier_out_2:
            secondary_produced_carrier: 0.5 # produces 0.5 units of ``secondary_produced_carrier`` for every 1 unit of ``primary_produced_carrier``
        carrier_out_3:
            tertiary_produced_carrier: 0.9 # produces 0.9 units of ``tertiary_produced_carrier`` for every 1 unit of ``primary_produced_carrier``

Where multiple carriers are included in a carrier level, any of those carriers can meet the carrier level requirement. They are listed in the same indented level, for example:

.. code-block:: yaml

    tech_identifier_1:
        primary_carrier: 'primary_produced_carrier' # ``primary_produced_carrier`` will be used to cost/constraint application
        carrier_in:
            primary_consumed_carrier: 1 # if chosen, will consume 1 unit of ``primary_consumed_carrier`` to meet the requirements of ``carrier_in``
            primary_consumed_carrier_2: 0.5 # if chosen, will consume 0.5 units of ``primary_consumed_carrier_2`` to meet the requirements of ``carrier_in``
        carrier_in_2:
            secondary_consumed_carrier: 0.8 # if chosen, will consume 0.8 units of ``secondary_consumed_carrier`` for every 1 unit of ``carrier_in`` being consumed
            secondary_consumed_carrier_2: 0.1 # if chosen, will consume 0.1 / 0.8 = 0.125 units of ``secondary_consumed_carrier_2`` for every 1 unit of ``carrier_in`` being consumed
        carrier_out:
            primary_produced_carrier: 1 # if chosen, will produce 1 unit of ``primary_produced_carrier`` for every 1 unit of ``carrier_out`` being produced
            primary_produced_carrier_2: 0.8 # if chosen, will produce 0.8 units of ``primary_produced_carrier_2`` for every 1 unit of ``carrier_in`` being produced

.. Note:: A ``primary_carrier`` must be defined when there are multiple ``carrier_out`` values defined. ``primary_carrier`` can be defined as any carrier in a technology's output carriers (including secondary and tertiary carriers).

``stack_weight`` and ``color`` determine how the technology is shown in model outputs. The higher the ``stack_weight``, the lower a technology will be shown in stackplots.

The ``depreciation`` definition is optional and only necessary if defaults need to be overridden. However, at least one constraint (such as ``e_cap.max``) and one cost should usually be defined.

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

.. Note::  Transmission technologies can define both an ``e_loss`` (per-distance) and an ``e_eff`` (distance-independent). For example, setting ``e_eff`` to 0.9 implies a 10% loss during transmission, independent of distance. If both ``e_loss`` and ``e_eff`` are defined, their effects are cumulative.

.. _config_reference_constraints:

Technology constraints
----------------------

The following table lists all available technology constraint settings and their default values. All of these can be set by ``tech_identifier.constraints.constraint_name``, e.g. ``nuclear.constraints.e_cap.max``.

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

.. _abstract_base_tech_definitions:

Abstract base technologies
--------------------------

This lists all pre-defined abstract base technologies and the defaults they provide. Note that it is not possible to define a technology with the same identifier as one of the abstract base technologies. In addition to providing default values for some options, which abstract base technology a user-defined technology inherits from determines how Calliope treats the technology internally. This internal treatment means that only a subset of available constraints are used for each of the abstract base technologies.

supply
^^^^^^

.. literalinclude:: includes/basetech_supply.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints`:

.. code-block:: yaml

    stack_weight
    color
    carrier_out
    group
    x_map
    export
    constraints:
        r
        force_r
        r_unit
        r_area.min
        r_area.max
        r_area.equals
        r_area_per_e_cap
        e_prod
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
        export_cap
    costs:
        r_area
        e_cap
        om_frac
        om_fixed
        om_var
        om_fuel
        export
    depreciation:
        plant_life
        interest
    weight

supply_plus
^^^^^^^^^^^

.. literalinclude:: includes/basetech_supply_plus.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints`:

.. code-block:: yaml

    stack_weight
    color
    carrier_out
    group
    x_map
    export
    constraints:
        r
        force_r
        r_unit
        r_eff
        r_area.min
        r_area.max
        r_area.equals
        r_area_per_e_cap
        r_cap.min
        r_cap.max
        r_cap.equals
        r_cap_equals_e_cap
        r_scale
        r_scale_to_peak
        allow_r2
        r2_startup_only
        r2_eff
        r2_cap.min
        r2_cap.max
        r2_cap.equals
        r2_cap_follow
        r2_cap_follow_mode
        s_init
        s_cap.min
        s_cap.max
        s_cap.equals
        c_rate
        s_time.max
        s_loss
        e_prod
        p_eff
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
        export_cap
    costs:
        s_cap
        r_area
        r_cap
        r2_cap
        e_cap
        om_frac
        om_fixed
        om_var
        om_fuel
        om_r2
        export
    depreciation:
        plant_life
        interest
    weight

demand
^^^^^^

.. literalinclude:: includes/basetech_demand.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints`:

.. code-block:: yaml

    stack_weight
    color
    carrier_in
    group
    x_map
    export
    constraints:
        r
        force_r
        r_unit
        r_area.min
        r_area.max
        r_area.equals
        r_area_per_e_cap
        e_con
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
    costs:
        r_area
        e_cap
        om_frac
        om_fixed
        om_var
        export
    depreciation:
        plant_life
        interest
    weight

unmet_demand
^^^^^^^^^^^^

.. literalinclude:: includes/basetech_unmet_demand.yaml
   :language: yaml


There is also the option to include unmet demand as a "true" supply technology by making use of ``unmet_demand_as_supply_tech``:

.. literalinclude:: includes/basetech_unmet_demand_as_supply_tech.yaml
   :language: yaml

In either case, the additional available constraints are the same as found for the supply abstract base technology. However, it is generally not advised to edit any constraints pertaining to `unmet_demand`.

storage
^^^^^^^

.. Warning:: The default value provided by ``storage`` for `e_con`` should not be overridden.

.. literalinclude:: includes/basetech_storage.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints` :

.. code-block:: yaml

    stack_weight
    color
    carrier
    group
    x_map
    export
    constraints:
        e_prod
        s_init
        s_cap.min
        s_cap.max
        s_cap.equals
        c_rate
        s_time.max
        s_loss
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
        export_cap
    costs:
        s_cap
        e_cap
        om_frac
        om_fixed
        om_var
        export
    depreciation:
        plant_life
        interest
    weight

transmission
^^^^^^^^^^^^

.. Warning:: The default value provided by ``transmission`` for``e_con`` should not be overridden.

.. literalinclude:: includes/basetech_transmission.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints` :

.. code-block:: yaml

    stack_weight
    color
    carrier
    group
    x_map
    export
    constraints:
        e_prod
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
        export_cap
    costs:
        e_cap
        om_frac
        om_fixed
        om_var
        export
    costs_per_distance:
        e_cap
    constraints_per_distance:
        e_loss
    depreciation:
        plant_life
        interest
    weight

conversion
^^^^^^^^^^

.. literalinclude:: includes/basetech_conversion.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints` :

.. code-block:: yaml

    stack_weight
    color
    carrier_in
    carrier_out
    group
    x_map
    export
    constraints:
        e_prod
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
        export_cap
    costs:
        e_cap
        om_frac
        om_fixed
        om_var
        export
    depreciation:
        plant_life
        interest
    weight


conversion_plus
^^^^^^^^^^^^^^^

.. literalinclude:: includes/basetech_conversion_plus.yaml
   :language: yaml

Available constraints are as follows, with full descriptions found above, in :ref:`config_reference_constraints` :

.. code-block:: yaml

    stack_weight
    color
    primary_carrier
    carrier_in
    carrier_in_2
    carrier_in_3
    carrier_out
    carrier_out_2
    carrier_out_3
    group
    x_map
    export
    constraints:
        e_prod
        e_eff
        e_cap.min
        e_cap.max
        e_cap.equals
        e_cap.total_max
        e_cap.total_equals
        e_cap_scale
        e_cap_min_use
        e_ramping
        export_cap
    costs:
        e_cap
        om_frac
        om_fixed
        om_var
        export
    depreciation:
        plant_life
        interest
    weight

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
   * ``output.format``:  Format to save results in, either ``netcdf`` or ``csv``
* ``parallel``: Settings used to generate parallel runs, see :ref:`run_config_parallel_runs` for the available options
* ``time``: Settings to adjust time resolution, see :ref:`run_time_res` for the available options
* ``override``: Override arbitrary settings from the model configuration. E.g., this could specify ``techs.nuclear.costs.monetary.e_cap: 1000`` to set the ``e_cap`` costs of ``nuclear``, overriding whatever was set in the model configuration
* ``model_override``: Path to a YAML configuration file which contains additional overrides for the model configuration. If both this and ``override`` are specified, anything defined in ``override`` takes precedence over model configuration added in the ``model_override`` file.
* ``solver_options``: A list of options, which are passed on to the chosen solver, and are therefore solver-dependent (see below)

.. _debugging_runs_config:

Debugging failing runs
^^^^^^^^^^^^^^^^^^^^^^

A number of run settings exist to make debugging failing runs easier:

* ``subset_y``, ``subset_x``, ``subset_t``: specify if only a subset of technologies (y), locations (x), or timesteps (t) should be used for this run. This can be useful for debugging purposes. The timestep subset can be specified as ``[startdate, enddate]``, e.g. ``['2005-01-01', '2005-01-31']``. The subsets are processed before building the model and applying time resolution adjustments, so time resolution functions will only see the reduced set of data.

In addition, settings relevant to debugging can be specified inside a ``debug`` block as follows:

* ``debug.keep_temp_files``: Whether to keep temporary files inside a ``Logs`` directory rather than deleting them after completing the model run (which is the default). Useful to debug model problems.
* ``debug.overwrite_temp_files``: When ``debug.keep_temp_files`` is true, and the ``Logs`` directory already exists, Calliope will stop with an error, but if this setting is true, it will overwrite the existing temporary files.
* ``debug.symbolic_solver_labels``: By default, Pyomo uses short random names for all generated model components, rather than the variable and parameter names used in the model setup. This is faster but for debugging purposes models must be human-readable. Thus, particularly when using ``debug.keep_temp_files: true``, this setting should also be set to ``true``.
* ``debug.echo_solver_log``: Displays output from the solver on screen while solving the model (by default, output is only logged to the log file, which is removed unless ``debug.keep_temp_files`` is true).

The following example debug block would keep temporary files, removing possibly existing files from a previous run beforehand:

.. code-block:: yaml

   debug:
       keep_temp_files: true
       overwrite_temp_files: true

Solver options
^^^^^^^^^^^^^^

Gurobi: Refer to the `Gurobi manual <https://www.gurobi.com/documentation/>`_, which contains a list of parameters. Simply use the names given in the documentation (e.g. "NumericFocus" to set the numerical focus value). For example:

.. code-block:: yaml

    solver: gurobi

    solver_options:
        Threads: 3
        NumericFocus: 2

CPLEX: Refer to the `CPLEX parameter list <https://www.ibm.com/support/knowledgecenter/en/SS9UKU_12.5.0/com.ibm.cplex.zos.help/Parameters/topics/introListAlpha.html>`_. Use the "Interactive" parameter names, replacing any spaces with underscores (for example, the memory reduction switch is called "emphasis memory", and thus becomes "emphasis_memory"). For example:

.. code-block:: yaml

    solver: cplex

    solver_options:
        mipgap: 0.01
        mip_polishafter_absmipgap: 0.1
        emphasis_mip: 1
        mip_cuts: 2
        mip_cuts_cliques: 3