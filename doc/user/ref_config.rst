-----------------------
Configuration reference
-----------------------

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

See the `YAML website <https://yaml.org/>`_ for more general information about YAML.

Calliope internally represents the configuration as :class:`~calliope.core.attrdict.AttrDict`\ s, which are a subclass of the built-in Python dictionary data type (``dict``) with added functionality such as YAML reading/writing and attribute access to keys.
