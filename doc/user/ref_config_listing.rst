--------------------------------
Listing of configuration options
--------------------------------

To run a model, two things are needed: a *model definition* that defines such things as technologies, locations, costs and constraints, and *run settings*, which specify how the given model should be run. At their most basic, these two components can be specified in just two YAML files:

* ``model.yaml``, which sets up the model and may import any number of additional files in order to split large models up into manageable units. It must also specify, via the ``data_path`` setting, the directory with data files for those technologies that have data explicit in space and time. The data directory must contain, at a minimum, a file called ``set_t.csv`` which defines the model's timesteps. See :ref:`configuration_timeseries` below for more information on this.
* ``run.yaml``, which sets up run-specific and environment-specific settings such as which solver to use. It must also, with the ``model`` setting, specify which model should be run, by pointing to that model's primary model configuration file (e.g., ``model.yaml``).

Either of these files can have an arbitrary name, but it makes sense to call them something like ``run.yaml`` (for the run settings) and ``model.yaml`` (for the model definition).

The remainder of this section deals with the model configuration. See :doc:`run_configuration` for the run configuration.

The model definition can be split into several files in two ways:

1. Model configuration files can can use an ``import`` statement to specify a list of paths to additional files to import (the imported files, in turn, may include further files, so arbitrary degrees of nested configurations are possible). The ``import`` statement can either give an absolute path or a path relative to the importing file. If a setting is defined both in the importing file and the imported file, the imported settings are overridden.

2. The ``model`` setting in the run settings may either give a single file or a list of files, which will be combined on model initialization. An example of this is:

.. code-block:: yaml

   model:
       - model.yaml  # Define general model settings
       - techs.yaml   # Define technologies, their constraints and costs
       - locations.yaml  # Define locations and transmission capacities

.. Note::

   Calliope includes a command-line tool, ``calliope new``, which will create a new model at the given path, based on the built-in national-scale example model and its run configuration::

      calliope new my_new_model

   This makes it easier to experiment with the built-in example, and to quickly create a model by working off an existing skeleton.

.. _configuration_techs:

Technologies
------------

A technology's identifier can be any alphanumeric string. The index of all technologies ``y`` is constructed at model instantiation from all defined technologies. At the very minimum, a technology should define some constraints and some costs. A typical supply technology that has an infinite resource without spatial or temporal definition might define:

.. code-block:: yaml

   my_tech:
      parent: 'supply'
      name: 'My test technology'
      carrier_out: 'some_energy_carrier'
      constraints:
         e_cap.max: 1000  # kW
      costs:
         monetary:
            e_cap: 500  # per kW of e_cap.max

A demand technology, with its demand data stored in a time series in the file ``demand.csv``, might look like this:

.. code-block:: yaml

   my_demand_tech:
      parent: 'demand'
      carrier_in: 'some_energy_carrier'
      constraints:
         r: 'file=demand.csv'

Technologies must always define a parent, and this can either be one of the pre-defined abstract base technologies or another previously defined technology. The pre-defined abstract base technologies that can be inherited from are:

* ``supply``: Supplies energy to a carrier, has a positive resource.
* ``supply_plus``: Supplies energy to a carrier, has a positive resource. Additional possible constraints, including efficiencies and storage, distinguish this from ``supply``.
* ``demand``: Demands energy from a carrier, has a negative resource.
* ``unmet_demand``: Supplies unlimited energy to a carrier with a very high cost, but does not get counted as a supply technology for analysis and grouping purposes. An ``unmet_demand`` technology for all relevant carriers should usually be included in a model to keep the solution feasible in all cases (see the :doc:`tutorials <tutorials>` for a practical example).
* ``unmet_demand_as_supply_tech``: Works like ``unmet_demand`` but is a normal ``supply`` technology, so it does get counted as a supply technology for analysis and grouping purposes.
* ``storage``: Stores energy.
* ``transmission``: Transmits energy from one location to another.
* ``conversion``: Converts energy from one carrier to another.
* ``conversion_plus``: Converts energy from one or more carrier(s) to one or more different carrier(s).

A technology inherits the configuration that its parent specifies (which, in turn, inherits from its own parent). The abstract base technologies inherit from a model-wide default technology called ``defaults``.

It is possible, for example, to define a ``wind`` technology that specifies generic characteristics for wind power plants, and then multiple additional technologies, such as ``wind_onshore`` and ``wind_offshore``, that specify ``parent: wind``, but also override some of the generic wind settings with their own.

See :ref:`overriding_tech_options` below for additional information on how technology settings propagate through the model and how they can be overridden.

Refer to :ref:`config_reference_techs` for a complete list of all available technology constraints and costs.

.. Note::

   The identifiers of the abstract base technologies are reserved and cannot be used for a user-defined technology. In addition, ``defaults`` is also a reserved identifier and cannot be used.

.. _config_parents_and_groups:

Parents and groups
^^^^^^^^^^^^^^^^^^

Because each technology must define a ``parent``, the definition of all technologies represents a tree structure, with the built-in defaults representing the root node, the built-in abstract base technologies inheriting from that root node, and all other user-defined technologies inheriting from one of the abstract base technologies.

There are two important aspects to this model definition structure.

First, only leaf nodes (the outermost nodes) in this tree may actually be used as technologies in model definitions. In other words, the parent-child inheritance structure allows technologies to inherit settings from their parents, but only those technologies without any children themselves are considered "real". Calliope will raise an error if this requirement is not met.

Second, every non-leaf node is implicitly a group of technologies, and the solution returned by Calliope reports aggregated information for each defined technology and its children (see :doc:`analysis`).

The ``group`` option only has an effect on supply diversity functionality in the analysis module (again, see :doc:`analysis` for details). Because every non-leaf technology is implicitly a group, those that should be considered as distinct groups for the purpose of diversity of supply must be explicitly marked with ``group: true``.


.. figure:: images/inheritance.*
   :alt: Technology inheritance tree

   An example of a simple technology inheritance tree. ``renewables`` could define any defaults that both ``pv`` and ``wind`` should inherit, furthermore, it sets ``group: true``. Thus, for purposes of supply diversity, ``pv`` and ``wind`` will be counted together, while ``nuclear`` will be counted separately.

Locations
---------

A location's name can be any alphanumeric string, but using integers makes it easier to define constraints for a whole range of locations by using the syntax ``from--to``. Locations can be given as a single location (e.g., ``location1``), a range of integer location names using the ``--`` operator (e.g., ``0--10``), or a comma-separated list of alphanumeric location names (e.g., ``location1,location2,10,11,12``). Using ``override``, some settings can be overridden on a per-location and per-technology basis (see below).

Locations may also define a parent location using ``within``, as shown in the following example:

.. code-block:: yaml

   locations:
       location1:
           techs: ['demand_power', 'nuclear']
           override:
               nuclear:
                   constraints:
                       e_cap.max: 10000
       location2:
           techs: ['demand_power']
       offshore1, offshore2:
           within: location2
           techs: ['offshore_wind']

The energy balancing constraint looks at a location's level to decide which locations to consider in balancing supply and demand. Locations that are not ``within`` another location are implicitly at the topmost level. Supply and demand within locations on the topmost level must always be be balanced, but they can exchange energy with each other via transmission technologies, which may define parameters such as costs, distance, and losses.

Locations that are contained within a parent location have implicit loss-free and cost-free transmission between themselves and the parent location. The balancing constraint makes sure that supply and demand within a location and its direct children is balanced.

.. Warning::

   If a location contained within a parent location itself defines children, it is no longer included in the implicit free transmission between its siblings and parent location. In turn, it receives implicit free transmission with its own children.

Transmission nodes
^^^^^^^^^^^^^^^^^^

A location can also act as just a branch in a transmission network. This is relevant for locations where transmission links split into several lines, without any other technologies at those locations. In this case, the location definition becomes:

.. code-block:: yaml

    locations:
          location1:
              techs: ['transmission-tech']

Where ``transmission-tech`` can refer to any previously defined ``transmission`` technology which passes through that location. Listing transmission technologies is not necessary for any other location type.

.. _transmission_links:

Transmission links
------------------

Transmission links are defined in the model definition as follows:

.. code-block:: yaml

   links:
      location1,location2:
         transmission-tech:
            constraints:
               e_cap.max: 10000
      location1,location3:
         transmission-tech:
            # ...
         another-transmission-tech:
            # ...

``transmission-tech`` can refer to any previously defined technology, but that technology must have the abstract base technology ``transmission`` as a parent

It is possible to specify multiple possible transmission technologies (e.g., with different costs or efficiencies) between two locations by simply listing them all.

Transmission links can also specify a distance, which transmission technologies can use to compute distance-dependent costs or efficiencies. An ``e_loss`` can be specified under ``constraints_per_distance`` and costs for any cost class can be specified under ``costs_per_distance`` (see example below).

.. code-block:: yaml

   links:
      location1,location2:
         transmission-tech:
            distance: 500

   techs:
      transmission-tech:
         # per_distance constraints specified per 100 units of distance
         per_distance: 100
         constraints_per_distance:
            e_loss: 0.01  # loss per 100 units of distance
         costs_per_distance:
            monetary:
               e_cap: 10  # cost per 100 units of distance

.. Note::

    Transmission links are bidirectional by default. To force unidirectionality for a given technology along a given link, you have to set the ``one_way`` constraint in the constraint definition of that technology, in that link:

    .. code-block:: yaml

        links:
          location1,location2:
             transmission-tech:
                constraints:
                    one_way: true

    This will only allow transmission from ``location1`` to ``location2``. To swap the direction, the link name must be inverted, i.e. ``location2,location1``.

.. _overriding_tech_options:

Overriding technology options
-----------------------------

Technologies can define generic options, for example ``name``, constraints, for example ``constraints.e_cap.max``, and costs, for example ``costs.monetary.e_cap``.

These options can be overridden in several ways, and whenever such an option is accessed by Calliope it works its way through the following list until it finds a definition (so entries further up in this list take precedence over those further down):

1. Override for a specific location ``x1`` and technology ``y1``, which may be defined via ``locations`` (e.g. ``locations.x1.override.y1.constraints.e_cap.max``)
2. Setting specific to the technology ``y1`` if defined in ``techs`` (e.g. ``techs.y1.constraints.e_cap.max``)
3. Check whether the immediate parent of the technology ``y`` defines the option (assuming that ``y1`` specifies ``parent: my_parent_tech``, e.g. ``techs.my_parent_tech.constraints.e_cap.max``)
4. If the option is still not found, continue along the chain of parent-child relationships. Since every technology should inherit from one of the abstract base technologies, and those in turn inherit from the model-wide defaults, this will ultimately lead to the model-wide default setting if it has not been specified anywhere else. See :ref:`config_reference_constraints` for a complete listing of those defaults.

The following technology options can be overriden on a per-location basis:

* ``x_map``
* ``constraints.*``
* ``constraints_per_distance.*``
* ``costs.*``

The following settings cannot be overridden on a per-location basis:

* Any other options, such as ``parent`` or ``carrier``
* ``costs_per_distance.*``
* ``depreciation.*``

.. _configuration_timeseries:

Using time series data
----------------------

.. Note::

   If a parameter is not explicit in time and space, it can be specified as a single value in the model definition (or, using location-specific overrides, be made spatially explicit). This applies both to parameters that never vary through time (for example, cost of installed capacity) and for those that may be time-varying (for example, a technology's available resource).

Defining a model's time steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Irrespective of whether it actually uses time-varying parameters, a model must at specify its timesteps with a file called ``set_t.csv``. This must contain two columns (comma-separated), the first one being integer indices, and the second, ISO 8601 compatible timestamps (usually in the format ``YYYY-MM-DD hh:mm:ss``, e.g. ``2005-01-01 00:00:00``).

For example, the first few lines of a file specifying hourly timesteps for the year 2005 would look like this:

.. code-block:: text

   0,2005-01-01 00:00:00
   1,2005-01-01 01:00:00
   2,2005-01-01 02:00:00
   3,2005-01-01 03:00:00
   4,2005-01-01 04:00:00
   5,2005-01-01 05:00:00
   6,2005-01-01 06:00:00

Defining time-varying parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For parameters that vary in time, time series data can be read from CSV files. This can be done in two ways (using the example of ``r``):

1. Specify ``r: file=filename.csv`` to pick the desired CSV file.
2. Specify ``r: file``. In this case, the file name is automatically determined according to the format ``tech_param.csv`` (e.g., ``pv_r.csv`` for the parameter ``r`` of a technology with the identifier ``pv``).

Each CSV file must have integer indices in the first column which match the integer indices from ``set_t.csv``. The first row must be column names, while the rest of the cells are the actual (integer or floating point) data values:

.. code-block:: text

   ,loc1,loc2,loc3,...
   0,10,20,10.0,...
   1,11,19,9.9,...
   2,12,18,9.8,...
   ...

In the most straightforward case, the column names in the CSV files correspond to the location names defined in the model (in the above example, ``loc1``, ``loc2`` and ``loc3``). However, it is possible to define a mapping of column names to locations. For example, if our model has two locations, ``uk`` and ``germany``, but the electricity demand data columns are ``loc1``, ``loc2`` and ``loc3``, then the following ``x_map`` definition will read the demand data for the desired locations from the specified columns:

.. code-block:: yaml

   electricity_demand:
      x_map: 'uk: loc1, germany: loc2'
      constraints:
         r: 'file=demand.csv'

.. Warning::

   After reading a CSV file, if any columns are missing (i.e. if a file does not contain columns for all locations defined in the current model), the value for those locations is simply set to :math:`0` for all timesteps.

.. Note::

   ``x_map`` maps column names in an input CSV file to locations defined in the model, in the format ``name_in_model: name_in_file``, with as many comma-separated such definitions as necessary.

In all cases, all CSV files, alongside ``set_t.csv``, must be inside the data directory specified by ``data_path`` in the model definition.

For example, the files for a model specified in ``model.yaml``, which defined ``data_path: model_data``, might look like this (``+`` are directories, ``-`` files):

.. code-block:: text

   - model.yaml
   + model_data/
      - set_t.csv
      - tech1_r.csv
      - tech2_r.csv
      - tech2_e_eff.csv
      - ...

When reading time series, the ``r_scale_to_peak`` option can be useful. Specifying this will automatically scale the time series so that the peak matches the given value. In the case of ``r`` for demand technologies, where ``r`` will be negative, the peak is instead a trough, and this is handled automatically. In the below example, the electricity demand timeseries is loaded from ``demand.csv`` and scaled such that the demand peak is 60,000:

.. code-block:: yaml

   electricity_demand:
      constraints:
         r: 'file=demand.csv'
         r_scale_to_peak: -60000

Calliope provides functionality to automatically adjust the resolution of time series data to make models more computationally tractable. See :ref:`run_time_res` for details on this.

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

Mandatory model-wide settings with no default values (see :doc:`ref_model_config` for more information on defining ``techs``, ``locations`` and ``links``):

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
   :header: Setting,Default,Name,Unit,Comments
   :widths: 10, 5, 10, 5, 15
   :stub-columns: 0

.. _config_reference_costs:

Technology costs
----------------

These are all the available costs, which are set to :math:`0` by default for every defined cost class. Costs are set by ``tech_identifier.costs.cost_class.cost_name``, e.g. ``nuclear.costs.monetary.e_cap``.

.. csv-table::
   :file: includes/default_costs.csv
   :header: Setting,Default,Name,Unit,Comments
   :widths: 10, 5, 10, 5, 15
   :stub-columns: 0

Technology depreciation settings apply when calculating levelized costs. The interest rate and life times must be set for each technology with investment costs.

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
* ``parallel``: Settings used to generate parallel runs, see :ref:`run_config_generate` for the available options
* ``time``: Settings to adjust time resolution, see :ref:`run_time_res` for the available options
* ``override``: Override arbitrary settings from the model configuration. E.g., this could specify ``techs.nuclear.costs.monetary.e_cap: 1000`` to set the ``e_cap`` costs of ``nuclear``, overriding whatever was set in the model configuration
* ``model_override``: Path to a YAML configuration file which contains additional overrides for the model configuration. If both this and ``override`` are specified, anything defined in ``override`` takes precedence over model configuration added in the ``model_override`` file.
* ``solver_options``: A list of options, which are passed on to the chosen solver, and are therefore solver-dependent (see below)

.. _debugging_runs_config:

Debugging failing runs
----------------------

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

.. Note::

   If using Calliope interactively in a Python session and/or developing custom constraints and analysis functionality, we recommend reading up on the `Python debugger <https://docs.python.org/3/library/pdb.html>`_ and (if using IPython or Jupyter Notebooks) making heavy use of the `%debug magic <https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug>`_.


.. _solver_options:

Solver options
--------------

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


.. _run_time_res:

Time resolution adjustment
--------------------------

Models must have a default timestep length (defined implicitly by the timesteps defined in ``set_t.csv``), and all time series files used in a given model must conform to that timestep length requirement.

However, this default resolution can be adjusted over parts of the dataset via configuring ``time`` in the run settings. At its most basic, this allows running a function that can perform arbitrary adjustments to the time series data in the model, via ``time.function``, and/or applying a series of masks to the time series data to select specific areas of interest, such as periods of maximum or minimum production by certain technologies, via ``time.masks``.

The available options include:

1. Uniform time resolution reduction through the resample function, which takes a `pandas-compatible rule describing the target resolution <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html>`_. For example, to resample to 6-hourly timesteps:

.. code-block:: yaml

   time:
       function: resample
       function_options: {'resolution': '6H'}

2. Deriving representative days from the input time series, by applying either k-means or hierarchical clustering as defined in :mod:`calliope.time_clustering`, for example:

.. code-block:: yaml

   time:
       function: apply_clustering
       function_options: {clustering_func: 'get_clusters_kmeans', how: 'mean', k: 20}

3. Heuristic selection: application of one or more of the masks defined in :mod:`calliope.time_masks`, via a list of masks given in ``time.masks``. See :ref:`api_time_masks` in the API documentation for the available masking functions. Options can be passed to the masking functions by specifying ``options``. A ``time.function`` can still be specified and will be applied to the masked areas (i.e. those areas of the time series not selected), as in this example which looks for the week of minimum and maximum potential wind production (assuming a ``wind`` technology was specified), then reduces the rest of the input time series to 6-hourly resolution:

.. code-block:: yaml

   time:
      masks:
          - {function: week, options: {day_func: 'extreme', tech: 'wind', how: 'max'}}
          - {function: week, options: {day_func: 'extreme', tech: 'wind', how: 'min'}}
      function: resample
      function_options: {'resolution': '6H'}

.. Note::

  When loading a model, all time steps initially have the same weight. Time step resolution reduction methods may adjust the weight of individual timesteps; this is used for example to give appropriate weight to the operational costs of aggregated typical days in comparison to individual extreme days, if both exist in the same processed time series. See the implementation of constraints in :mod:`calliope.constraints.base` for more detail.

.. _run_config_generate:

Overrides and generating scripts for running multiple scenarios
---------------------------------------------------------------

* Create a ``run.yaml`` file with a ``parallel:`` section as needed (see :ref:`run_config_generate`).
* On the command line, run ``calliope generate path/to/run.yaml``.
* By default, this will create a new subdirectory inside a ``runs`` directory in the current working directory. You can optionally specify a different target directory by passing a second path to ``calliope generate``, e.g. ``calliope generate path/to/run.yaml path/to/my_run_files``.
* Calliope generates several files and directories in the target path. The most important are the ``Runs`` subdirectory which hosts the self-contained configuration for the runs and ``run.sh`` script, which is responsible for executing each run. In order to execute these runs in parallel on a compute cluster, a submit.sh script is also generated containing job control data, and which can be submitted via a cluster controller (e.g., ``qsub submit.sh``).

The ``run.sh`` script can simply be called with an integer argument from the sequence (1, number of parallel runs) to execute a given run, e.g. ``run.sh 1``, ``run.sh 2``, etc. This way the runs can easily be executed irrespective of the parallel computing environment available.

.. Note:: Models generated via ``calliope generate`` automatically save results as a single NetCDF file per run inside the parallel runs' ``Output`` subdirectory, regardless of whether the ``output.path`` or ``output.format`` options have been set.

The run settings can also include a ``parallel`` section.

This section is parsed when using the ``calliope generate`` command-line tool to generate a set of runs to be executed in parallel. A run settings file defining ``parallel`` can still be used to execute a single model run, in which case the ``parallel`` section is simply ignored.

The concept behind parallel runs is to specify a base model (via the run configuration's ``model`` setting), then define a set of model runs using this base model, but overriding one or a small number of settings in each run. For example, one could explore a range of costs of a specific technology and how this affects the result.

Specifying these iterations is not (yet) automated, they must be manually entered under ``parallel.iterations:`` section. However, Calliope provides functionality to gather and process the results from a set of parallel runs (see :doc:`analysis`).

At a minimum, the ``parallel`` block must define:

* a ``name`` for the run
* the ``environment`` of the cluster (if it is to be run on a cluster), currently supported is ``bsub`` and ``qsub``. In either case, the generated scripts can also be run manually
* ``iterations``: a list of model runs, with each entry giving the settings that should be overridden for that run. The settings are *run settings*, so, for example, ``time.function`` can be overridden. Because the run settings can themselves override model settings, via ``override``, model settings can be specified here, e.g. ``override.techs.nuclear.costs.monetary.e_cap``.

The following example parallel settings show the available options. In this case, two iterations are defined, and each of them overrides the nuclear ``e_cap`` costs (``override.techs.nuclear.costs.monetary.e_cap``):

.. code-block:: yaml

   parallel:
       name: 'example-model'  # Name of this run
       environment: 'bsub'  # Cluster environment, choices: bsub, qsub
       data_path_adjustment: '../../../model_config'
       # Execute additional commands in the run script before starting the model
       pre_run: ['source activate pyomo']
       # Execute additional commands after running the model
       post_run: []
       iterations:
           - override.techs.nuclear.costs.monetary.e_cap: 1000
           - override.techs.nuclear.costs.monetary.e_cap: 2000
       resources:
           threads: 1  # Set to request a non-default number of threads
           wall_time: 30  # Set to request a non-default run time in minutes
           memory: 1000  # Set to request a non-default amount of memory in MB

This also shows the optional settings available:

* ``data_path_adjustment``: replaces the ``data_path`` setting in the model configuration during parallel runs only
* ``pre_run`` and ``post_run``: one or multiple lines (given as a list) that will be executed in the run script before / after running the model. If running on a computing cluster, ``pre_run`` is likely to include a line or two setting up any environment variables and activating the necessary Python environment.
* ``resources``: specifying these will include resource requests to the cluster controller into the generated run scripts. ``threads``, ``wall_time``, and ``memory`` are available. Whether and how these actually get processed or honored depends on the setup of the cluster environment.

For an iteration to override more than one setting at a time, the notation is as follows:

.. code-block:: yaml

   iterations:
       - first_option: 500
         second_option: 10
       - first_option: 600
         second_option: 20
