
===================
Model configuration
===================

.. Note::

   See :doc:`configuration_reference` for a complete listing of all available configuration options.

To run a model, two things are needed: a *model definition* (also referred to as the *model settings*) that define such things as technologies, locations, costs and constraints, and *run settings*, which specify how the given model should be run. At its most basic, these two components can be specified in just two YAML files:

* ``model.yaml`` which sets up the model and may import any number of additional files in order to split large models up into manageable units. It mus also specify, via the ``data_path:`` directive, the directory with data files for those technologies that have data explicit in space and time. The data directory must contain, at a minimum, a file called ``set_t.csv`` which defines the model's timesteps. See :ref:`configuration_timeseries` below for more information on this.
* ``run.yaml`` which sets up run-specific and environment-specific settings such as which solver to use. It must also, with the ``model:`` directive, specify which model should be run by pointing to that model's primary model configuration file (e.g., ``model.yaml``).

Either of these files can have an arbitrary name, but for consistency we will refer to them as ``run.yaml`` (for the run settings) and ``model.yaml`` (for the model definition).

The remainder of this section deals with the model configuration, see :doc:`run_configuration` for the run configuration.

There are two ways to split the model definition can be split into several files:

1. Model configuration files can can use an ``import:`` statement to specify a list of paths to additional files to import (the imported files, in turn, may include further files, so arbitrary degrees of nested configurations are possible). The ``import:`` statement can either give an absolute path or a path relative to the importing file. If a setting is defined both in the importing file and the imported file, the imported settings are overridden.

2. The ``model:`` directive in the run settings may either give a single file or a list of files with, which will be combined in on model initialization (so may not define the same setting twice). An example of this is:

.. code-block:: yaml

   model:
       - model.yaml  # Define general model settings
       - techs.yaml   # Define technologies, their constraints and costs
       - locations.yaml  # Define locations and transmission capacities

.. Note::

   Calliope includes a command-line tool, ``calliope new``, which will create a new model based on the built-in example model at the given path, e.g.::

      calliope new models/my_new_model

   This makes it easier to quickly create a new model based on an existing skeletion.

.. _configuration_techs:

------------
Technologies
------------

A technology's identifier can be any alphanumeric string. The index of all technologies ``y`` is constructed at model instantiation from all defined technologies. At the very minimum, a technology should define some constraints and some costs. A typical supply technology that has an infinite resource without spatial or temporal definition might define:

.. code-block:: yaml

   my_tech:
      parent: 'supply'
      name: 'My test technology'
      constraints:
         e_cap_max: 1000  # kW
         r: inf
      costs:
         monetary:
            e_cap: 500  # per kW of e_cap_max

A demand technology based on a time series in the file ``demand_test.csv`` might define:

.. code-block:: yaml

   consumption-tech:
      parent: 'demand'
      constraints:
         r: 'file=demand_test.csv'

Technologies must always define a parent, and this can either be one of the pre-defined abstract base technologies or another previously defined technology. The pre-defined abstract base technologies that can be inherited from are:

* ``supply``: Supplies energy to a carrier, has a positive resource
* ``demand``: Demands energy from a carrier, has a negative resource
* ``unmet_demand``: Supplies unlimited energy to a carrier with a very high cost, but does not get counted as a supply technology for analysis and grouping purposes. An ``unmet_demand`` technology for all relevant carriers should usually be included in a model to keep the solution feasible in all cases (see the :doc:`tutorial <tutorial>` for a practical example).
* ``unmet_demand_as_supply_tech``: Works like ``unmet_demand`` but is a normal ``supply`` technology, so it does get counted as a supply technology for analysis and grouping purposes
* ``storage``: Stores energy
* ``transmission``: Transmits energy from one location to another
* ``conversion``: Converts energy from one carrier to another

A technology inherits the configuration that its parent specifies (which, in turn, inherits from its own parent). The abstract base technologies inherit from a model-wide default technology called ``defaults``.

It is possible, for example, to define a ``wind`` technology that specifies generic characteristics for wind power plants, and then multiple additional technologies, such as ``wind_onshore`` and ``wind_offshore``, that specify ``parent: wind``, but also override some of the generic wind settings with their own.

See :ref:`overriding_tech_options` below for additional information on how technology settings propagate through the model and how they can be overridden.

Refer to :ref:`config_reference_techs` for a complete list of all available technology constraints and costs.

.. Note::

   The identifiers of the abstract base technologies are reserved and cannot be used for a user-defined technology. In addition, ``defaults`` is also a reserved identifier and cannot be used.

---------
Locations
---------

A location's name can be any alphanumeric string, but using integers makes it easier to define constraints for a whole range of locations by using the syntax ``from--to``. The index of all locations ``x`` is constructed at model instantiation from all locations defined in the configuration.

There are currently some limitations to how locations work:

* Locations must be assigned to either level 0 or level 1 (``level:``).
* Locations at level 0 may be assigned to a parent location from level 1 (``within:``).
* Using ``override:``, some (but not all) settings can be overriden on a per-location and per-technology basis (see the box below).

Locations can be given as a single location (e.g., ``location1``), a range of integer location names using the ``--`` operator (e.g., ``0--10``), or a comma-separated list of location names (e.g., ``location1,location2,10,11,12``).

An example locations block is:

.. code-block:: yaml

   locations:
       location1:
           level: 1
           techs: ['demand_power', 'nuclear']
           override:
               nuclear:
                   constraints:
                       e_cap_max: 10000
       location2:
           level: 1
           techs: ['demand_power']
       offshore1:
           level: 0
           within: location2
           techs: ['offshore_wind']
       offshore2:
            level: 0
            within: location2
            techs: ['offshore_wind']


.. Note::

   *Only* the following constraints can be overriden on a per-location and per-tech basis (for now). Attempting to override any others will cause errors or simply be ignored:

   * x_map
   * constraints: r, r_eff, e_eff, c_eff, r_scale, r_scale_to_peak, s_cap_max, s_cap_max_force, s_init, s_time, s_time_max, use_s_time, r_cap_max, r_area_max, e_cap_max, e_cap_max_scale, e_cap_max_force, rb_eff, rb_cap_max, rb_cap_max_force, rb_cap_follows,

.. NB this limitation is "implemented" simply by calling get_option with an x=x argument for some options but not for others

The balancing constraint looks at a location's level to decide which locations to consider in balancing supply and demand. Currently, balancing of supply and demand takes place between locations at level 1. In order for a location at level 0 to be included in the system-wide energy balance, it must therefore be assigned to a parent location at level 1. Transmission is *loss-free* within a location, between locations at level 0, and from locations at level 0 to locations at level 1. Transmission is only possible between locations at level 1 if a transmission link has been defined between them (see below). Losses in these transmission links are as defined for the specified transmission technology.

.. Warning::

   There must always be at least one location at level 1, because balancing of supply and demand takes place between level 1 locations only (this will be improved in the future).

.. _transmission_links:

------------------
Transmission links
------------------

Transmission links are defined in the model settings as follows:

.. code-block:: yaml

   links:
      location1,location2:
         transmission-tech:
            constraints:
               e_cap_max: 10000
      location1,location3:
         transmission-tech:
            # ...
         another-transmission-tech:
            # ...

``transmission-tech`` can refer to any previously defined technology, but that technology must have the abstract base technology ``transmission`` as a parent

It is possible to specify multiple possible transmission technologies (e.g., with different costs or efficiencies) between two locations by simply listing them all.

Transmission links can also specify a distance, which transmission technologies can use to compute distance-dependent costs or efficiencies. An ``e_loss`` can be specified under ``constraints_per_distance`` and any costs and cost classes can be specified under ``costs_per_distance`` (see example below).

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

.. _overriding_tech_options:

-----------------------------
Overriding technology options
-----------------------------

Technologies can define generic options, for example, ``name``, constraints, for example ``constraints.e_cap_max``, and costs, for example ``costs.monetary.e_cap``.

These options can be overridden in several ways, and whenever such an option is accessed by Calliope it works its way through the following list until it finds a definition (so an upper entry in this list take precedence over a lower entry):

1. Override for a specific location ``x1`` and technology ``y1``, which may be defined in the ``locations:`` directive (e.g. ``locations.x1.override.y1.constraints.e_cap_max``)
2. Setting specific to the technology ``y1`` if defined in ``techs:`` directive (e.g. ``techs.y1.constraints.e_cap_max``)
3. Check whether the immediate parent of the technology ``y`` defines the option (assuming that ``y1`` specifies ``parent: my_parent_tech``, e.g. ``techs.my_parent_tech.constraints.e_cap_max``)
4. If the option is still not found, continue along the chain of parent relationships. Since every technology should inherit from one of the abstract base technologies, and those in turn inherit from the model-wide defaults, this will ultimately lead to the model-wide default setting if it has not been specified anywhere else. See :ref:`config_reference_constraints` for a complete listing of those defaults.

.. _configuration_timeseries:

----------------------
Using time series data
----------------------

If a parameter is not explicit in time and space, it can be simply specified in the model settings (and, using location-specific overrides, be made spatially explicit).

Each model however must at a minimum specify all timesteps with a file called ``set_t.csv``. This must contain two columns (comma-separated), the first one being integer indices, and the second, ISO 8601 compatible timestamps (usually in the format ``YYYY-MM-DD hh:mm:ss``, e.g. ``2005-01-01 00:00:00``).

For example, the first few lines of a file specifying hourly timesteps for the year 2005 would look like this:

.. code-block:: text

   0,2005-01-01 00:00:00
   1,2005-01-01 01:00:00
   2,2005-01-01 02:00:00
   3,2005-01-01 03:00:00
   4,2005-01-01 04:00:00
   5,2005-01-01 05:00:00
   6,2005-01-01 06:00:00

Time series data can be used to specify the ``r`` and ``e_eff`` parameters for specific technologies. This can be done in two ways (using the example of ``r``):

1. Specify ``r: file=filename.csv`` to pick the desired CSV file.
2. Specify ``r: file``. In this case, the file name is automatically determined according the format ``tech_param.csv`` (e.g., ``pv_r.csv`` for the parameter ``r`` of a technology with the identifier ``pv``).

Each CSV file must have integer indices in the first column which match the integer indices from ``set_t.csv``. The first row must be column names, while the rest of the cells are the actual (integer or floating point) data values:

.. code-block:: text

   ,loc1,loc2,loc3,...
   0,10,20,10.0,...
   1,11,19,9.9,...
   2,12,18,9.8,...
   ...

In the most straightforward case, the column names in the CSV files correspond to the location names defined in the model (in the above example, ``loc1``, ``loc2`` and ``loc3``). However, it is possible to define a mapping of column names to locations. For example, if our model has two locations, ``uk`` and ``germany``, but the electricity demand data columns are ``loc1``, ..., then the following ``x_map`` definition will properly read the demand data for the desired locations:

.. code-block:: yaml

   electricity_demand:
      x_map: 'uk: loc1, germany: loc2'
      constraints:
         r: 'file=demand.csv'

.. Warning::

   After reading a CSV file, if any columns are missing (i.e. if a file does not contain columns for all locations defined in the current model), the value for those locations is simply set to :math:`0` for all timesteps.

In all cases, all CSV files, alongside ``set_t.csv``, must be inside the data directory specified by ``data_path:`` in the model settings.

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

.. _loading_optional_constraints:

----------------------------
Loading optional constraints
----------------------------

Additional constraints can be loaded in ``model.yaml`` by specifying ``constraints``, for example:

.. code-block:: yaml

   contraints:
       - constraints.ramping.ramping_rate
       - my_custom_module.my_constraint

When resolving constraint names, Calliope first checks whether the constraint is part of Calliope itself (in the above example, this is the case for ``constraints.ramping.ramping_rate``, which is included in Calliope). If the constraint is not found as part of Calliope, the first part of the dot-separated name is interpreted as a Python module name (in the above example, ``my_custom_module``). The module is imported and the constraint loaded from it.

This architecture makes it possible to add constraints in a modular way without modifying the Calliope source code. Custom constraints have access to all model configuration, so that additional settings can easily be included anywhere in the model configuration to support the functionality of custom constraints. See :doc:`develop` for information on this.
