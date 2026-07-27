# Examples & tutorials

In this section of the documentation, we will take you through some simple and more advanced topics related to building and running Calliope models in Python.
These examples are best understood after you have gone through the [basic concepts](../getting_started/concepts.md).

Some of the tutorials are based on the built-in example models.
They explain the key steps necessary to set up and run simple models.
The built-in examples are simple on purpose, to show the key components of a Calliope model with which models of arbitrary complexity can be built.

<div class="grid cards" markdown>

-   :fontawesome-solid-shapes:{ .lg .middle } __Math gallery__

    ---

    * [Math gallery](../user_defined_math/examples/index.md): reusable, user-defined math for implementing advanced constraints.

-   :fontawesome-solid-diagram-project:{ .lg .middle } __Example models__

    ---

    * [National scale](national_scale/index.md): part of a national grid, using supply with and without a storage buffer, a storage technology, and inheriting from technology and node groups.
    * [Urban scale](urban_scale/index.md): part of a district network, using conversion technologies with single and multiple output carriers, revenue generation through carrier export, and inheriting from templates.
    * [MILP](milp/index.md): extends the urban scale model with binary and integer decision variables (extending an LP model to a MILP model).

-   :fontawesome-brands-python:{ .lg .middle } __Python tutorials__

    ---

    * [Loading tabular data](loading_tabular_data.py): define your model directly from tabular data.
    * [Running in different modes](modes.py): build and solve in `plan`, `operate`, and `spores` modes.
    * [Piecewise linear constraints](piecewise_constraints.py): add piecewise linear constraints to a model.
    * [The model and backend objects](calliope_model_object.py): explore the `calliope.Model` and `Model.backend` objects.
    * [Logging](calliope_logging.py): capture and direct Calliope's logging output.

</div>
