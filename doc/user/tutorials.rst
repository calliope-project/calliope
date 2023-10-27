=========
Tutorials
=========

The tutorials are based on the built-in example models, they explain the key steps necessary to set up and run simple models. Refer to the other parts of the documentation for more detailed information on configuring and running more complex models. The built-in examples are simple on purpose, to show the key components of a Calliope model with which models of arbitrary complexity can be built.

The :doc:`first tutorial <tutorials_01_national>` builds a model for part of a national grid, exhibiting the following Calliope functionality:

* Use of supply, supply_plus, demand, storage and transmission technologies
* Nested locations
* Multiple cost types

The :doc:`second tutorial <tutorials_02_urban>` builds a model for part of a district network, exhibiting the following Calliope functionality:

* Use of supply, demand, conversion, conversion_plus, and transmission technologies
* Use of multiple carriers
* Revenue generation, by carrier export

The :doc:`third tutorial <tutorials_03_milp>` extends the second tutorial, exhibiting binary and integer decision variable functionality (extended an LP model to a MILP model)

.. toctree::
   :maxdepth: 2

   tutorials_01_national
   tutorials_02_urban
   tutorials_03_milp
