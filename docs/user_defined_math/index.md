# Defining your own math

As of Calliope version 0.7, all of the internal math used to build optimisation problems is stored in YAML files.

The same syntax used for the [pre-defined math](https://github.com/calliope-project/calliope/tree/main/src/calliope/math) can be used to define additional math.
So, if you want to introduce new constraints, decision variables, or objectives, you can do so as part of the collection of YAML files describing your model.

In brief, components of the math formulation are stored under named keys and contain information on:

* the sets over which they will be generated (e.g., for each technology, node, timestep, ...),
* the conditions under which they will be built in any specific model (e.g., if no `storage` technologies exist in the model, decision variables and constraints associated with them will not be built), and
* their math expression(s).

In this section, we will describe the [math components](components.md) and the [formulation syntax](syntax.md) in more detail.
Whenever we refer to a "math component" it could be a:

* decision variable (something you want the optimisation model to decide on the value of).
* global expression (a mixture of decision variables and input parameters glued together with math).
* constraint (a way to limit the upper/lower bound of a decision variable using other decision variables/parameters/global expressions).
* objective (the expression whose value you want to minimise/maximise in the optimisation).

We recommend you first review the available [math components](components.md),
then go into detail on the math formulation [syntax](syntax.md),
before looking at how to [add your own math](customise.md) and browsing the [gallery of user-defined math examples](../user_defined_math/examples/index.md).
A full reference for the allowed key-value pairs in a math YAML file is available in the [reference section of the documentation][model-math-schema].

!!! note

    Although we have tried to make a generalised syntax for all kinds of user-defined math, our focus was on building a system to implement Calliope's base math.
    Therefore, we cannot guarantee that your math will be possible to implement.

!!! warning

    When writing your own math, remember that Calliope is a _linear_ modelling framework.
    It is possible that your desired math will create a nonlinear optimisation problem.
    Usually, the solver will provide a clear error message when this is the case, although it may not be straightforward to pinpoint what part of your math is the culprit.
