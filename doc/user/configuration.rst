
===================
Model configuration
===================

.. Note::

   See :doc:`configuration_reference` for a complete listing of all available configuration options.

To run a model, two things are needed: a *model definition* that defines such things as technologies, locations, costs and constraints, and *run settings*, which specify how the given model should be run. At their most basic, these two components can be specified in just two YAML files:

* ``model.yaml``, which sets up the model and may import any number of additional files in order to split large models up into manageable units. It must also specify, via the ``data_path`` setting, the directory with data files for those technologies that have data explicit in space and time. The data directory must contain, at a minimum, a file called ``set_t.csv`` which defines the model's timesteps. See :ref:`configuration_timeseries` below for more information on this.
* ``run.yaml``, which sets up run-specific and environment-specific settings such as which solver to use. It must also, with the ``model`` setting, specify which model should be run, by pointing to that model's primary model configuration file (e.g., ``model.yaml``).

Either of these files can have an arbitrary name, but it makes sense to call them something like ``run.yaml`` (for the run settings) and ``model.yaml`` (for the model definition).

The remainder of this section deals with the model configuration, see :doc:`run_configuration` for the run configuration.

There are two ways by the model definition can be split into several files:

1. Model configuration files can can use an ``import`` statement to specify a list of paths to additional files to import (the imported files, in turn, may include further files, so arbitrary degrees of nested configurations are possible). The ``import`` statement can either give an absolute path or a path relative to the importing file. If a setting is defined both in the importing file and the imported file, the imported settings are overridden.

2. The ``model`` setting in the run settings may either give a single file or a list of files, which will combined on model initialization. An example of this is:

.. code-block:: yaml

   model:
       - model.yaml  # Define general model settings
       - techs.yaml   # Define technologies, their constraints and costs
       - locations.yaml  # Define locations and transmission capacities

.. Note::

   Calliope includes a command-line tool, ``calliope new``, which will create a new model at the given path, based on the built-in example model and its run configuration::

      calliope new my_new_model

   This makes it easier to experiment with the built-in example, and to quickly create a model by working off an existing skeleton.

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
         e_cap.max: 1000  # kW
         r: inf
      costs:
         monetary:
            e_cap: 500  # per kW of e_cap.max

A demand technology, with its demand data stored in a time series in the file ``demand.csv``, might look like this:

.. code-block:: yaml

   my_demand_tech:
      parent: 'demand'
      constraints:
         r: 'file=demand.csv'

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

.. _config_parents_and_groups:

Parents and groups
------------------

Because each technology must define a ``parent``, the definition of all technologies represents a tree structure, with the built-in defaults representing the root node, the built-in abstract base technologies inheriting from that root node, and all other user-defined technologies inheriting from one of the abstract base technologies.

There are two important aspects to this model definition structure.

First, only leaf nodes (the outermost nodes) in this tree may actually be used as technologies in model definitions. In other words, the parent-child inheritance structure allows technologies to inherit settings from their parents, but only those technologies without any children themselves are considered "real". Calliope will raise an error if this requirement is not met.

Second, every non-leaf node is implicitly a group of technologies, and the solution returned by Calliope reports aggregated information for each defined technology and its children (see :doc:`analysis`).

The ``group`` option only has an effect on supply diversity functionality in the analysis module (again, see :doc:`analysis` for details). Because every non-leaf technology is implicitly a group, those that should be considered as distinct groups for the purpose of diversity of supply must be explicitly marked with ``group: true``.


.. figure:: images/inheritance.*
   :alt: Technology inheritance tree

   An example of a simple technology inheritance tree. ``renewables`` could define any defaults that both ``pv`` and ``wind`` should inherit, furthermore, it sets ``group: true``. Thus, for purposes of supply diversity, ``pv`` and ``wind`` will be counted together, while ``nuclear`` will be counted separately.

---------
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

.. _transmission_links:

------------------
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

.. _overriding_tech_options:

-----------------------------
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

----------------------
Using time series data
----------------------

If a parameter is not explicit in time and space, it can be simply specified as a single value in the model definition (or, using location-specific overrides, be made spatially explicit).

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
2. Specify ``r: file``. In this case, the file name is automatically determined according to the format ``tech_param.csv`` (e.g., ``pv_r.csv`` for the parameter ``r`` of a technology with the identifier ``pv``).

.. Note:: If ``e_eff`` is loaded as a time series from file, ``e_eff_ref`` (a static reference efficiency) has to be specified, which is needed for example to compute storage capacity if a storage time is given.

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

.. _loading_optional_constraints:

----------------------------
Loading optional constraints
----------------------------

Calliope uses "constraint generator" functions that read the model configuration and build model constraints based on it. Constraint generators for optional constraints are included in the :mod:`calliope.constraints.optional` module. In addition, custom-built user constraints can be added by loading additional constraint generator functions. They can be added in ``model.yaml`` by specifying ``constraints``, for example:

.. code-block:: yaml

   constraints:
       - constraints.optional.ramping_rate
       - my_custom_module.my_constraint

When resolving constraint names, Calliope first checks whether the constraint is part of Calliope itself (in the above example, this is the case for ``constraints.optional.ramping_rate``, which is included in Calliope). If the constraint is not found as part of Calliope, the first part of the dot-separated name is interpreted as a Python module name (in the above example, ``my_custom_module``). The module is imported and the constraint loaded from it.

This architecture makes it possible to add constraints in a modular way without modifying the Calliope source code. Custom constraints have access to all model configuration, so that additional settings can easily be included anywhere in the model configuration to support the functionality of custom constraints. See :doc:`develop` for information on this.
