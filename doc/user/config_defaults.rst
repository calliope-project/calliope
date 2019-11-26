--------------------------
Configuration and defaults
--------------------------

This section lists the available configuration options and constraints along with their default values.
Defaults are automatically applied in constraints whenever there is no user input for a particular value.

.. _config_reference_model:

Model configuration
-------------------

.. csv-table::
   :file: includes/model_settings.csv
   :header: Setting,Default,Comments
   :widths: 10, 5, 15
   :stub-columns: 0

.. _config_reference_run:

Run configuration
-----------------

.. csv-table::
   :file: includes/run_settings.csv
   :header: Setting,Default,Comments
   :widths: 10, 5, 15
   :stub-columns: 0

.. _config_reference_constraints:

Per-tech constraints
--------------------

The following table lists all available technology constraint settings and their default values. All of these can be set by :yaml:`tech_identifier.constraints.constraint_name`, e.g. :yaml:`nuclear.constraints.energy_cap.max`.

.. csv-table::
   :file: includes/default_constraints.csv
   :header: Setting,Default,Name,Unit,Comments
   :widths: 10, 5, 10, 5, 15
   :stub-columns: 0

.. _config_reference_costs:

Per-tech costs
--------------

These are all the available costs, which are set to :math:`0` by default for every defined cost class. Costs are set by :yaml:`tech_identifier.costs.cost_class.cost_name`, e.g. :yaml:`nuclear.costs.monetary.energy_cap`.

.. csv-table::
   :file: includes/default_costs.csv
   :header: Setting,Default,Name,Unit,Comments
   :widths: 10, 5, 10, 5, 15
   :stub-columns: 0

Technology depreciation settings apply when calculating levelized costs. The interest rate and life times must be set for each technology with investment costs.

Group constraints
-----------------

See :ref:`group_constraints` for a full listing of available group constraints.

.. _abstract_base_tech_definitions:

Abstract base technology groups
-------------------------------

Technologies must always define a parent, and this can either be one of the pre-defined abstract base technology groups or a user-defined group (see :ref:`tech_groups`). The pre-defined groups are:

* ``supply``: Supplies energy to a carrier, has a positive resource.
* ``supply_plus``: Supplies energy to a carrier, has a positive resource. Additional possible constraints, including efficiencies and storage, distinguish this from ``supply``.
* ``demand``: Demands energy from a carrier, has a negative resource.
* ``storage``: Stores energy.
* ``transmission``: Transmits energy from one location to another.
* ``conversion``: Converts energy from one carrier to another.
* ``conversion_plus``: Converts energy from one or more carrier(s) to one or more different carrier(s).

A technology inherits the configuration that its parent group specifies (which, in turn, may inherit from its own parent).

.. Note::

   The identifiers of the abstract base tech groups are reserved and cannot be used for user-defined technologies. However, you can amend an abstract base technology group for example by a lifetime attribute that will be in effect for all technologies derived from that group (see :ref:`tech_groups`).

.. _defaults:

The following lists the pre-defined base tech groups and the defaults they provide.

supply
^^^^^^

.. figure:: images/supply.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_supply.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_supply.yaml
   :language: yaml

supply_plus
^^^^^^^^^^^

.. figure:: images/supply_plus.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_supply_plus.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_supply_plus.yaml
   :language: yaml

demand
^^^^^^

.. figure:: images/demand.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_demand.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_demand.yaml
   :language: yaml

storage
^^^^^^^

.. figure:: images/storage.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_storage.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_storage.yaml
   :language: yaml

transmission
^^^^^^^^^^^^

.. figure:: images/transmission.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_transmission.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_transmission.yaml
   :language: yaml

conversion
^^^^^^^^^^

.. figure:: images/conversion.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_conversion.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_conversion.yaml
   :language: yaml

conversion_plus
^^^^^^^^^^^^^^^

.. figure:: images/conversion_plus.*

Default constraints provided by the parent tech group:

.. literalinclude:: includes/basetech_conversion_plus.yaml
   :language: yaml

Required constraints, allowed constraints, and allowed costs:

.. literalinclude:: includes/required_allowed_conversion_plus.yaml
   :language: yaml
