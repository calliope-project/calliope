--------------------------------
Listing of configuration options
--------------------------------

.. _config_reference_model_wide:

Configuration layout
--------------------

There must always be at least one model configuration YAML file, probably called ``model.yaml`` or similar. This file can import any number of additional files.

This file or this set of files must specify the following top-level configuration keys:

* ``name``: the name of the model
* ``model``: model settings
* ``run``: run settings
* ``techs``: technology definitions
* (optionally) ``tech_groups``: tech group definitions
* ``locations``: location definitions
* (optionally) ``links``: transmission link definitions

.. Note:: Model settings (``model``) affect how the model and its data are built by Calliope, while run settings (``run``) only take effect once a built model is run (e.g. interactively via ``model.run()``). This means that run settings, unlike model settings, can be updated after a model is built and before it is run, by modifying attributes in the built model dataset.

.. _config_reference_model:

List of model settings
----------------------

.. csv-table::
   :file: includes/model_settings.csv
   :header: Setting,Default,Comments
   :widths: 10, 5, 15
   :stub-columns: 0

.. _config_reference_run:

List of run settings
--------------------

.. csv-table::
   :file: includes/run_settings.csv
   :header: Setting,Default,Comments
   :widths: 10, 5, 15
   :stub-columns: 0

.. _config_reference_constraints:

List of possible constraints
----------------------------

The following table lists all available technology constraint settings and their default values. All of these can be set by ``tech_identifier.constraints.constraint_name``, e.g. ``nuclear.constraints.e_cap.max``.

.. csv-table::
   :file: includes/default_constraints.csv
   :header: Setting,Default,Name,Unit,Comments
   :widths: 10, 5, 10, 5, 15
   :stub-columns: 0

.. _config_reference_costs:

List of possible costs
----------------------

These are all the available costs, which are set to :math:`0` by default for every defined cost class. Costs are set by ``tech_identifier.costs.cost_class.cost_name``, e.g. ``nuclear.costs.monetary.e_cap``.

.. csv-table::
   :file: includes/default_costs.csv
   :header: Setting,Default,Name,Unit,Comments
   :widths: 10, 5, 10, 5, 15
   :stub-columns: 0

Technology depreciation settings apply when calculating levelized costs. The interest rate and life times must be set for each technology with investment costs.

.. _abstract_base_tech_definitions:

List of abstract base technology groups
---------------------------------------

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

   The identifiers of the abstract base tech groups are reserved and cannot be used for a user-defined technology or tech group.

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

.. _yaml_format:

YAML configuration file format
------------------------------

All configuration files (with the exception of time series data files) are in the YAML format, "a human friendly data serialisation standard for all programming languages".

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

Calliope internally represents the configuration as :class:`~calliope.core.attrdict.AttrDict`\ s, which are a subclass of the built-in Python dictionary data type (``dict``) with added functionality such as YAML reading/writing and attribute access to keys.
