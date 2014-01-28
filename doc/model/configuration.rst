
==================================
Model definition and configuration
==================================

A model run consists of the *run settings* and the associated *model definition* (also referred to as the *model settings*). At its most basic, these two components are specified in just two YAML files:

* ``run.yaml`` which sets up run-specific and environment-specific settings such as which solver to use. It must also, under ``input:``, define at least two directives:
   1. the ``path:`` directive giving the directory with data files defining parameters explicitly in space and time, which must contain at the very minimum a file ``set_t.csv`` (see :doc:`data`)
   2. another directive with any name, usually ``model:``, giving the path to another YAML file containing the model definition (``model.yaml``).
* ``model.yaml`` which sets up the model and may import any number of additional files in order to split large models up into manageable units.

Either of these files can have an arbitrary name, but for consistency we will refer to them as ``run.yaml`` (for the run settings) and ``model.yaml`` (for the model definition).

There are two ways to split the model definition into several files:

1. The ``input:`` directive in the run settings may give any number of additional files with any name (the only setting which has a fixed name is ``path:``). These will be loaded in the order in which they are specified. An example of this is:

.. code-block:: yaml

   input:
      path: /path/to/data/directory
      model: model.yaml  # Define general model settings
      techs: techs.yaml   # Define technologies, their constraints and costs
      locations: locations.yaml  # Define locations and transmission capacities

2. Any of the model configuration files given in the ``input:`` directive may contain an ``import:`` statement, which is a list of paths to additional files to import. These additional files may each again contain an ``import:`` statement, allowing for arbitrarily complex nesting. The ``import:`` statement can either give an absolute path, a path relative to the importing file, or a path starting with ``{{ module }}``, which is a placeholder for the Calliope module's location (and thus useful to load included default configurations).

------------
Run settings
------------

The run settings take care of setting up and running the model with a given model configuration. It can be operated in two modes:

1. Directly instantiate an instance of :class:`~calliope.Model` and run it by calling its ``run()`` method. This ignores any settings in the ``parallel`` block in the run settings.
2. Set up a series of parallel runs via the ``calliope_run.py`` command-line tool (:ref:`explained here <parallel_runs>`). This will result in a set of scripts to perform the desired model runs either locally or on a remote cluster.

In case (1), the run settings file can be specified by setting ``config_run`` when instantiating the Model, e.g. ``calliope.Model(config_run='/path/to/run.yaml')``. In case (2), the run settings file is a required argument to ``calliope_run.py``.

---------------------
Defining technologies
---------------------

A technology's name can be any alphanumeric string. The index of all technologies ``y`` is constructed at model instantiation from all defined technologies. At the very minimum, a technology should define some constraints and some costs. For a production technology, it should define:

.. code-block:: yaml

   supply-tech:
      parent: 'default'
      constraints:
         e_cap_max: ...
         r: ...
      costs:
         e_cap: ...

A consumption technology should define:

.. code-block:: yaml

   consumption-tech:
      parent: 'demand'
      constraints:
         r: ...

------------------
Defining locations
------------------

A location's name can be any alphanumeric string, but using integers makes it easier to define constraints for a whole range of locations by using the syntax ``from--to``. The index of all locations ``x`` is constructed at model instantiation from all locations defined in the configuration.

There are currently some limitations to how locations work:

* Locations must be assigned to either level 0 or level 1 (``level:``).
* Locations may be assigned to a parent location (``within:``).
* Using ``override:``, specific settings can be overriden on a per-location and per-technology basis.

Locations can be given as a single location (e.g., ``location0``), a range of integer location names using the ``--`` operator (e.g., ``0--10``), or a comma-separated list of location names (e.g., ``location0,location1,10,11,12``).

.. admonition:: Note

   *Only* the following constraints can be overriden on a per-location and per-tech basis (for now). Attempting to override any others will cause errors or simply be ignored:

   * x_map
   * constraints: r, r_eff, e_eff, r_scale_to_peak, s_cap_max, s_init, r_cap_max, r_area_max, e_cap_max, e_cap_max_force

All locations are created equal, but the balancing constraint looks at a location's level to decide which locations to consider in balancing supply and demand. Currently, balancing of supply and demand takes place at level 1 only. In order for a location at level 0 to be included in the system-wide energy balance, it must therefore be assigned to a parent location at level 1. Transmission is *loss-free* within a location, between locations at level 0, and from locations at level 0 to locations at level 1. Transmission is only possible between locations at level 1 if a transmission link has been defined between them. Losses in these transmission links are as defined for the specified transmission technology.

.. admonition:: Note

   There must always be at least one location at level 1, because balancing of supply and demand takes place at level 1 locations only (this will be improved in a future version).

Transmission links
==================

Transmission links are defined in the model settings as follows:

.. code-block:: yaml

   links:
      location0,location1:
         transmission-tech:
            constraints:
               ...
      location1,location2:
         transmission-tech:
            ...
         another-tranmisssion-tech:
            ...

``transmission-tech`` can be any technology, but a useful transmission technology must define ``r: inf, e_can_be_negative: true`` and specify an ``e_cap_max`` (see the definition for ``transmission`` in the example model's ``techs.yaml``). It is possible to specify any amount of possible tranmission technologies (for example with different costs or efficiencies) between two locations by simply listing them all with their constraints.

-----------
Inheritance
-----------

The model definition uses an inheritance chain that starts at the top and works its way through the following list until it finds a setting:

1. Override for a specific location ``x`` and technology ``y`` if defined in the ``locations:`` directive
2. Setting specific to technology ``y`` if defined in ``techs:`` directive
3. Starting with immediate parent of the technology ``y``, check across the chain of inheritance
4. The last technology at the top of the inheritance chain should define a parent ``defaults``, which is loaded from a technology called ``defaults`` defined ``defaults.yaml``

--------------------------------------
How parameters are read from CSV files
--------------------------------------

If a parameter is not explicit in time and space, it is simply read from the model settings as needed during model generation, using the ``get_option()`` method.

If a parameter is explicit in time and space, it is read and stored in the :class:`~calliope.Model` object's ``data`` attribute during its instantiation (in ``read_data()``).

There are various limitations in how this happens, which make some combinations of custom values difficult. However, it is always possible to modify them manually after model instantiation before calling ``generate_model()``.

The parameters this currently applies to are:

* ``r``
* ``r_eff``
* ``e_eff``

The steps taken for each of these parameters ``param``, for technology ``y``, are:

1. Load the parameters from the model settings for ``y`` (going through the inheritance chain to the ``defaults`` if needed). If a numerical value is given, it is stored (in ``read_data()``) and later set as the parameter value for all ``x, t`` (in ``generate_model()``).

2. If on the other hand ``file`` is given, try loading the parameter from a CSV file, with the format ``{y}_{param}.csv`` (for example ``pv_r.csv`` for a PV resource parameter). The CSV file must contain timesteps as rows and locations as columns.

.. admonition:: Note

   After reading the CSV file, if any columns are missing (i.e. if a file does not contain columns for all locations in the current :class:`~calliope.Model`'s locations set), they are added with a value of 0 for all timesteps.

---------------------
Specifying a CSV file
---------------------

Instead of letting Calliope look for CSV data files according to the default naming scheme (:doc:`data`), it is possible to manually specify a CSV file for a specific technology.

There are two ways to do this, with the first one usually being the preferred way:

1. Using ``file=filename`` it is possible to manually specify a file to be read (inside the model's data directory) on a per-technology, per-location basis:

.. code-block:: yaml

   demand:
      constraints:
         r: 'file=demand-eu_r.csv'
         r_scale_to_peak: -60000

2. Alternatively, it is possible to simply define an additional technology that inherits from the desired parent technology, but whose name matches with the desired data files. In the example below, the technology ``demand-eu`` would look for the data file ``demand-eu_r.csv`` without the need to further specify a filename:

.. code-block:: yaml

   demand-eu:
      r: file  # If `demand` does not already specify this
      parent: 'demand'

--------------------------
Settings for parallel runs
--------------------------

The run settings can (but do not have to) define a ``parallel:`` section. This section is parsed when using the ``calliope_run.py`` command-line tool to generate a set of runs to be run in parallel (:ref:`explained here <parallel_runs>`).

The available options are detailed in the example model's run settings (``run.yaml``).
