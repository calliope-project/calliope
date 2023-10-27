.. role:: green
.. role:: red
.. role:: yellow

Mathematical formulation
########################

This section details the mathematical formulation that is loaded when creating a Calliope model.
All built-in Calliope math can be found in the calliope `math directory <https://github.com/calliope-project/calliope/tree/main/calliope/math>`_.

By default, the :ref:`base math <base_math>` is loaded from file.
If you want to overwrite the base math with other built-in math, you can do so by referring to the file by its name (without the file extension) in :yaml:`model.custom_math`, e.g. :yaml:`model.custom_math: [storage_inter_cluster]`.
When solving the model in a :ref:`run mode <config_reference_config>` other than `plan`, some built-in custom math will be applied automatically from a file of the same name (e.g., `spores` mode custom math is stored in `math/spores.yaml <https://github.com/calliope-project/calliope/blob/main/calliope/math/spores.yaml>`_).
The changes made by the built-in custom math are detailed in this page.

.. note:: Custom math is applied in the order it appears in the :yaml:`model.custom_math` list. By default, any run mode custom math will be applied as the final step. If you want to apply your own custom math *after* the run mode custom math, you should add it explicitly to the :yaml:`model.custom_math` list, e.g., :yaml:`model.custom_math: [operate, my_custom_math.yaml]`.


A guide to the math documentation
=================================

If a math component's initial conditions are met (those to the left of the curly brace), it will be applied to a model.
For each objective, constraint and global expression, a number of subconditions then apply (those to the right of the curly brace) to decide on the specific expression to apply at a given iteration of the component dimensions.

In the following expressions, terms in **bold** font are decision variables and terms in *italic* font are parameters. A list of the decision variables is given at the end of this page. A detailed listing of parameters along with their units and default values is given  :doc:`here <config_defaults>`.
Those parameters which are defined over time (`timesteps`) in the expressions can be defined by a user as a single, time invariant value, or as a timeseries that is :ref:`loaded from file or dataframe <configuration_timeseries>`.


Writing your own math documentation
===================================

To view only the mathematical formulation valid for your own model, you can write a LaTeX or reStructuredText file as follows:

.. code-block:: python

    model = calliope.Model("path/to/model.yaml")
    model.build_math_documentation(include="valid")
    model.write_math_documentation(filename="path/to/output/file.[tex|rst]")

.. _base_math:

Base math
=========

.. include:: ../_static/math.rst

.. _storage_inter_cluster_math:

Inter-cluster storage custom math
=================================
Below are the changes from the base math introduced by the built-in custom math file ``storage_inter_cluster``.

.. include:: ../_static/math_storage_inter_cluster.rst

Operate mode custom math
========================
Below are the changes from the base math introduced by the built-in custom math file ``operate``.
These changes are applied automatically if selecting the run mode ``operate``

TODO: Add operate math

SPORES mode custom math
========================
Below are the changes from the base math introduced by the built-in custom math file ``spores``.
These changes are applied automatically if selecting the run mode ``spores``

TODO: Add spores math
