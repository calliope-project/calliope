.. role:: green
.. role:: red
.. role:: yellow

Mathematical formulation
========================

This section details the base mathematical formulation that is loaded when creating a Calliope model.
If a math component's initial conditions are met (those to the left of the curly brace), it will be applied to a model.
For each objective, constraint and global expression, a number of subconditions then apply (those to the right of the curly brace) to decide on the specific expression to apply at a given iteration of the component dimensions.

In the following expressions, terms in **bold** font are decision variables and terms in *italic* font are parameters. A list of the decision variables is given at the end of this page. A detailed listing of parameters along with their units and default values is given  :doc:`here <config_defaults>`.
Those parameters which are defined over time (`timesteps`) in the expressions can be defined by a user as a single, time invariant value, or as a timeseries that is :ref:`loaded from file or dataframe <configuration_timeseries>`.

To view only the mathematical formulation valid for your own model, you can write a LaTeX or reStructuredText file as follows:

.. code-block:: python

    model = calliope.Model("path/to/model.yaml")
    model.build_math_documentation(include="valid")
    model.write_math_documentation(filename="path/to/output/file.[tex|rst]")

Base math
---------

.. include:: ../_static/math.rst

Storage inter cluster custom math
---------------------------------
Below are the changes from the base math introduced by the inbuilt custom math file `storage_inter_cluster`.

.. include:: ../_static/math_storage_inter_cluster.rst


