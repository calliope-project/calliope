
=======================
Configuration reference
=======================

Model-wide settings
-------------------

TODO

.. code-block:: yaml

   # General parameters
   startup_time: 12  # Length of startup period (hours)

   # Operation mode settings
   opmode:
       horizon: 48  # Optimization period length (hours)
       window: 24  # Operation period length (hours)

   # Per-carrier system margins
   system_margin:
       power: 0

Technology
----------

A technology with the name ``tech_name`` is configured by a YAML block within a ``techs:`` block. The following block shows all available options and their defaults (see further below for the constraints, costs, and depreciation definitions):

.. code-block:: yaml

   tech_name:
       stack_weight: 100  # Weight of this technology in the stack when plotting
       color: false  # HTML color code, if false, a random one will be chosen
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

The ``depreciation`` definition is optional and only necessary if defaults need to be overridden. However, at least one constraint (such as ``e_cap_max``) and one cost should usually be defined.

Technology constraints
----------------------

.. csv-table::
   :file: includes/default_constraints.csv
   :header: Setting,Default,Details
   :widths: 10, 5, 30
   :stub-columns: 0

Technology costs
----------------

These are all the available costs, which are set to :math:`0` by default for every defined cost class:

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

``storage``:

.. literalinclude:: includes/basetech_storage.yaml
   :language: yaml

``transmission``:

.. literalinclude:: includes/basetech_transmission.yaml
   :language: yaml

``conversion``:

.. literalinclude:: includes/basetech_conversion.yaml
   :language: yaml
