
==================================
Model definition and configuration
==================================

A model run consists of the run settings (``run.yaml``) and the associated model definition (also referred to as the model configuration). The model definition consists of four sources:

* ``model.yaml``: defines general model settings
* ``techs.yaml``: defines all possible technologies, their constraints and costs
* ``nodes.yaml``: defines all nodes, groups them into regions, and defines possible transmission capacities
* the path to a folder with data files defining parameters explicitly in space and time, which must contain at the very minimum a file ``set_t.csv`` (see :doc:`data`)

The run settings take care of setting up and running the model with a given model configuration. It can be operated in two modes:

1. Directly instantiate an instance of :class:`~calliope.Model` and run it by calling its ``run()`` method. This ignores any settings in the ``parallel`` block in ``run.yaml``
2. Set up a series of parallel runs via the ``calliope_run.py`` command-line tool (:ref:`explained here <parallel_runs>`). This will result in a set of scripts to perform the desired model runs either locally or on a remote cluster.

In case (1), the ``run.yaml`` file can be specified by setting ``config_run`` when instantiating the Model, e.g. ``calliope.Model(config_run='/path/to/run.yaml')``. In case (2), the ``run.yaml`` file can is a required argument to ``calliope_run.py``. All other paths are set under the ``input`` option inside ``run.yaml``.

---------------------
Defining technologies
---------------------

Technologies are defined in ``techs.yaml``. A technology's name can be any alphanumeric string. The index of all technologies ``y`` is constructed at model instantiation from all defined technologies. At the very minimum, a technology should define some constraints and some costs. For a production technology, it should define:

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

--------------
Defining nodes
--------------

Nodes are defined in ``nodes.yaml``. A node's name can be any alphanumeric string, but using integers makes it easier to define constraints for a whole range of nodes by using the syntax ``from--to``. The index of all nodes ``x`` is constructed at model instantiation from all nodes defined in the configuration.

There are currently some limitations to how nodes work:

* Nodes must be assigned to either level 0 or level 1 (``level:``).
* Nodes may be assigned to a parent node (``within:``).
* Using ``override:``, specific settings can be overriden on a per-node and per-technology basis.

Nodes can be given as a single node (e.g., ``node0``), a range of integer node names using the ``--`` operator (e.g., ``0--10``), or a comma-separated list of node names (e.g., ``node0,node1,10,11,12``).

.. admonition:: Note

   *Only* the following constraints can be overriden on a per-node and per-tech basis (for now). Attempting to override any others will cause errors or simply be ignored:

   * x_map
   * constraints: r, r_eff, e_eff, r_scale_to_peak, s_cap_max, s_init, r_cap_max, r_area_max, e_cap_max

All nodes are created equal, but the balancing constraint looks at a node's level to decide which nodes to consider in balancing supply and demand. Currently, balancing of supply and demand takes place at level 1 only. In order for a node at level 0 to be included in the system-wide energy balance, it must therefore be assigned to a parent node at level 1. Transmission is *loss-free* within a node, between nodes at level 0, and from nodes at level 0 to nodes at level 1. Transmission is only possible between nodes at level 1 if a transmission link has been defined between them. Losses in these transmission links are as defined for the specified transmission technology.

.. admonition:: Note

   There must always be at least one node at level 1, because balancing of supply and demand takes place at level 1 nodes only (this will be improved in a future version).

Transmission links
==================

Transmission links are defined in the ``nodes.yaml`` file as follows:

.. code-block:: yaml

   links:
      node0,node1:
         transmission-tech:
            constraints:
               ...
      node1,node2:
         transmission-tech:
            ...
         another-tranmisssion-tech:
            ...

``transmission-tech`` can be any technology, but a useful transmission technology must define ``r: inf, e_can_be_negative: true`` and specify an ``e_cap_max`` (see the definition for ``transmission`` in the example model's ``techs.yaml``). It is possible to specify any amount of possible tranmission technologies (for example with different costs or efficiencies) between two nodes by simply listing them all with their constraints.

-----------
Inheritance
-----------

The model definition uses an inheritance chain that starts at the top and works its way through the following list until it finds a setting:

1. Override for a specific node ``x`` and technology ``y`` from ``nodes.yaml``
2. Setting specific to technology ``y`` defined in ``techs.yaml``
3. Starting with immediate parent of the technology ``y``, check across the chain of inheritance
4. The last technology at the top of the inheritance chain should define a parent ``defaults``, which is a special reference to the defaults defined in ``defaults.yaml`` across all technologies

-----------------------
How parameters are read
-----------------------

If a parameter is not explicit in time and space, it is simply read from ``model_settings.yaml`` as needed during model generation, using the ``get_option()`` method.

If a parameter is explicit in time and space, it is read and stored in the :class:`~calliope.Model` object's ``data`` attribute during its instantiation (in ``read_data()``).

There are various limitations in how this happens, which make some combinations of custom values difficult. However, it is possible to modify them manually after instantiation and before calling ``generate_model()``.

The parameters this currently applies to are:

* ``r``
* ``r_eff``
* ``e_eff``

The steps taken for each of these parameters ``param``, for technology ``y``, are:

1. Check if the parameter is defined in ``model_settings.yaml`` for ``y``. If so, the value is read and stored (in ``read_data()``) and later set as the parameter value for all ``x, t`` (in ``generate_model()``).

2. If (1) does not apply, try loading the parameter from a CSV file, with the format ``{y}_{param}.csv``, so for example ``pv_r.csv`` for a PV resource parameter. The CSV file must contain timesteps as rows and nodes as columns.

.. admonition:: Note

   After reading the CSV file, if any columns are missing (i.e. if a file does not contain columns for all nodes in the current :class:`~calliope.Model`'s nodes set), they are added with a value of 0 for all timesteps.


3. If neither (1) nor (2) apply, the value is read analogously to (1) but from the ``default`` technology in ``model_settings.yaml``.

``read_data()`` will fail if neither of these three steps work for a parameter.

---------------------
Specifying a CSV file
---------------------

Instead of letting Calliope look for CSV data files according to the default naming scheme (:doc:`data`), it is possible to manually specify a CSV file for a specific technology.

There are two ways to do this, with the first one usually being the preferred way:

1. Using ``file=filename`` it is possible to manually specify a file to be read (inside the model's data directory) on a per-technology, per-node basis:

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

The available options are detailed in the example model's ``run.yaml`` file.
