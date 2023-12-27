--------------------------
Configuration and defaults
--------------------------

This section lists the available configuration options and constraints along with their default values.
Defaults are automatically applied in constraints whenever there is no user input for a particular value.

.. _config_reference_config:

.. include:: includes/config_schema.md
    :parser: myst_parser.sphinx_

.. _abstract_base_tech_definitions:

Abstract base technology groups
-------------------------------

Technologies must always define a parent, which must be one of the pre-defined abstract base technology groups:

* ``supply``: Draws from a source to produce a carrier.
* ``demand``: Consumes a carrier to supply to an external sink.
* ``storage``: Stores a carrier.
* ``transmission``: Transmits a carrier from one location to another.
* ``conversion``: Converts a carrier from one to another.

A technology will have decision variables available and constraints applied to it according to its parent.

Inheriting configurations
-------------------------

To share definitions between technologies and/or nodes, you can use configuration inheritance (the `inherit` key).
This allows a technology/node to inherit definitions from `tech_group`/`node_group` definitions.

.. note::
   Inheritance is different to setting a `parent`.
   Setting a parent does not entail any configuration options being inherited.
   It is only used when building the optimisation problem (i.e., in the `math`).

Configuration defaults
----------------------

.. _config_reference_defaults:

.. include:: includes/model_def_schema.md
    :parser: myst_parser.sphinx_

