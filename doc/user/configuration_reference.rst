
=======================
Configuration reference
=======================

General YAML configuration file format
--------------------------------------

TODO

Discuss ``import:`` directive

Using '' or "" to mark strings is optional, but can help with readability.

Model-wide settings
-------------------

These settings can either be in a central ``model.yaml`` file, or imported from other files if desired.

Mandatory model-wide settings with no default values (example settings are shown here). See :doc:`configuration` for more information on defining ``techs``, ``locations`` and ``links``:

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
       # ... setup for group_fraction constraints (see configuration section) ...

   metadata:  # Metadata for analysis and plotting
       map_boundary: []
       location_coordinates:
       location_ordering:

.. TODO document metadata settings

Optional model-wide settings that have defaults set by Calliope (default values are shown here):

.. code-block:: yaml

   startup_time: 12  # Length of startup period (hours)

   opmode:  # Operation mode settings
       horizon: 48  # Optimization period length (hours)
       window: 24  # Operation period length (hours)

   system_margin:  # Per-carrier system margins
       power: 0

.. _config_reference_techs:

Technology
----------

A technology with the identifier ``tech_identifier`` is configured by a YAML block within a ``techs:`` block. The following block shows all available options and their defaults (see further below for the constraints, costs, and depreciation definitions):

.. code-block:: yaml

   tech_identifier:
       name:  # A descriptive name, e.g. "Offshore wind"
       parent:  # An abstract base technology, or a previously defined one
       stack_weight: 100  # Weight of this technology in the stack when plotting
       color: false  # HTML color code, or `false` to choose a random color
       source_carrier: false # Carrier to consume, for conversion technologies
       group: false  # Set to ``true`` if this is a group
       weight: 1.0 # Cost weighting in objective function
       constraints:
           # ... constraint definitions ...
       costs:
           monetary:
               # ... monetary cost definitions ...
           # ... other cost classes ...
       depreciation:
           # ... depreciation definitions ...

Each technology **must** define a ``parent``, which can either be an abstract base technology such as ``supply``, or any other technology previously defined in the model. The technology inherits all settings from its parent, but overwrites anything it specifies again itself.

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

Run settings
------------

.. TODO

TODO
