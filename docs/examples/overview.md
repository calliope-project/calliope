# Examples and tutorials

In this section of the documentation, we will take you through some simple and more advanced topics related to building and running Calliope models in Python.

Some of the tutorials are based on the built-in example models.
They explain the key steps necessary to set up and run simple models.
The built-in examples are simple on purpose, to show the key components of a Calliope model with which models of arbitrary complexity can be built.

The ["national scale" example](national_scale/index.md) builds a model for part of a national grid, exhibiting the following Calliope functionality:

* Use of supply with and without a storage buffer.
* Use of a storage technology.
* Inheriting from technology and node groups.

The ["urban scale" example](urban_scale/index.md) builds a model for part of a district network, exhibiting the following Calliope functionality:

* Use of conversion technologies with singular and multiple output carriers.
* Revenue generation, by carrier export.
* Inheriting from templates

The ["MILP" example](milp/index.md) extends the urban scale example, exhibiting binary and integer decision variable functionality (extended an LP model to a MILP model).
