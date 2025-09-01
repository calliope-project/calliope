# Basic concepts

This page explains the basic concepts and ideas behind Calliope.
We then move on to describing how to [create](creating.md), [run](running.md), and [analyse](analysing.md) a Calliope model.

!!! note
    The [examples and tutorials section](../examples/index.md) contains more hands-on examples of how to build and work with Calliope models. We still recommend that you first read the section you are currently looking at - "Getting started" - before going to the examples and tutorials.

## What Calliope does

Calliope is an energy system modelling framework based on mathematical optimisation.
It is designed to formulate and solve typical problems from the energy field such as:

* Capacity expansion planning
* Economic dispatch
* Power market modelling

It is used in such roles by both commercial and research organisations.
What sets Calliope apart from other tools is its focus on keeping even large models human-readable through the use of text-based model definitions, explained in more detail below.
Due to its high degree of customisability, Calliope is also particularly well suited for rapid prototyping and development.

## Mathematical modelling terminology

Some of the terminology in Calliope comes from the field of [operations research](https://en.wikipedia.org/wiki/Operations_research) (also called mathematical programming or mathematical modelling):

* **Parameters**: Numerical values which are fixed parts of the problem, i.e. user-supplied input data.
* **Variables**: Numerical values which are determined by Calliope as part of the model solving process.
* **Constraints**: Mathematical functions that define the model and give bounds to the values that the variables can take.
* **Objective function**: A mathematical function that is maximised or minimised to find values for the variables.

To dive into these concepts in more detail, you can refer to [Modelling Energy Systems](https://www.modelling-energy-systems.org/), a free online online reader based on a university course taught by one of the Calliope developers.

## Calliope terminology: how the world is represented in Calliope

A Calliope model is a collection of interconnected technologies, nodes and carriers describing a real world system of flows.
Calliope implements the mathematics for all of these, allowing you, the user, to concentrate on describing your system using the building blocks defined by Calliope.
Usually, we consider these flows to be _energy_ flows (or in the case of a power system model, _electricity_ flows).
Most of what you will read in this documentation concerns energy systems.
However, the concepts are just as applicable to other types of flows, such as water, or material goods.

These are the most important concepts around which Calliope's maths are built:

* **Carriers** are commodities whose flows we track, e.g., electricity, heat, hydrogen, water, CO<sub>2</sub>.
* **Technologies** supply, consume, convert, store or transmit _carriers_, e.g., transmission lines/pipes, batteries, power plants, wind turbines, or home appliances.
* **Nodes** contain groups of _technologies_ and are usually geographic, e.g., a country, municipality or a single house.
* Carrier flows can enter the system from **sources**, e.g., energy from the sun to power a solar panel, and can exit it into **sinks**, e.g., electricity consumed by household appliances.
Unlike _carriers_, we do not explicitly track the type of commodity described by sources and sinks.

The visual overview below gives you a sense of how a simple model might be set up.
It has two nodes (blue boxes), two carriers (yellow and red), and various technologies (grey boxes).
As we will see further below, all of these building blocks are supplied in Calliope, so all we have to do is specify how we want them to be wired together and provide data for them.

![Visual description of the Calliope terminology.](../img/description_of_system.svg)

Many of the model variables (e.g. the power output from a power plant) and parameters (e.g. the demand for electricity) vary through time.
To deal with this, Calliope also has a concept of time through discrete **timesteps**.
Thus, Calliope represents space as discrete nodes and time as discrete timesteps.

Putting all of these possibilities together allows a modeller to create a model that is as simple or complex as necessary to answer a given research question.
Calliope's syntax ensures these models are intuitive, and easy to understand and share.

## Building blocks of models in Calliope

### YAML: keys and values

Models in Calliope are defined in a text file format called YAML, referring to tabular data files (in the CSV format) where necessary.
These files are essentially a collection of `key: value` entries, where `key` is a given setting - for example the nameplate capacity of a power plant - and `value` might be a number, a text string, or a list (e.g. a list of further settings).
We will often refer to "keys" and "values" in the documentation.
The keys and values can be nested, for example:

```yaml
top_level_option:
  second_level_option:
    third_level_option_1: 10
    third_level_option_2: 20
```

You will see the term "top-level key" in the documentation: that means a key at the very top of the "hierarchy" defined by this nesting of configuration.

One important Calliope-specific feature is the ability to spread your model across as many YAML files as you want, and use `import` top-level keys to "glue" them together.
Typically, you will have a main model file (e.g., `model.yaml`), from which you import other files.

You will see many examples of YAML as you proceed through the documentation, and most of what is happening should be intuitively understandable.
However, if you want a more detailed and systematic description of how YAML is used in Calliope, you can refer to our [YAML reference](reference/yaml.md).

### Model configuration (including math) and model definition (data)

Within the YAML file(s) that define your model, we distinguish between model configuration and model definition:

* The model **configuration** are the options provided to Calliope to do its work, and this includes specifying what maths to use. Specifying what maths to use means specifying what kinds of model components will exist and how they will behave. The configuration is listed under the top-level key [`config`](config.md).

* The model **definition** is your representation of the physical system you are modelling and includes the data with which the components specified in the math will be "populated". It spans across the four top-level keys [`techs`](techs.md), [`nodes`](nodes.md), [`data_definitions`](parameters.md), and [`data_tables`](data_tables.md).

!!! note
    Later, once you start looking at Calliope model data from a successful model run, you will see three main types of numerical data, which are a mix of model inputs and outputs:

    * **`inputs`**: these correspond to parameters, i.e. your input data.
    * **`results`**: these correspond to variables, i.e. the solution found by Calliope when solving your model.
    * Some of the results are **post-processed results** data that Calliope calculates after the mathematical model is solved.
    For example, capacity factors are calculated in post-processing based on the operation of all technologies, but in a mathematical modelling sense, they are not variables in the model.

### Math: Base math, mode math, and extra math

The maths underlying a Calliope model is also defined in YAML files.
Calliope supplies built-in math and allows users to partially or fully modify and replace this math.

There is what we call the built-in **base math**.
It is called base because it is active by default.
By default, it defines a [capacity planning problem][base-math] with perfect foresight.
It includes, for example, the basic concepts of carriers, nodes, and techs described above.

On top of the base math, it is possible to activate **mode math**.
This allows special cases which require additional processing, for example, the operate (dispatch / receding horizon control) and SPORES (near-optimal alternative generation) modes.

Finally, it is possible to supply **extra math** which are applied on top of the base and mode math (if used).
For instance, the [inter-cluster storage][inter-cluster-storage-math] extra math supplied with Calliope allows you to track storage levels in technologies more accurately when you are using timeseries clustering in your model.

Calliope follows a strict order of priority when applying math: **base math -> mode math -> extra math**.

### Overrides and scenarios

The final two basic concepts to know about are **overrides** and **scenarios**. They are defined in the top-level YAML keys [`overrides` and `scenarios`](scenarios.md).

Their purpose is define alternatives to the model configuration/definition that you can refer to when you initialise your model.
For example, you might want to explore several pre-defined capacity expansion plans in a model of the European power grid.
To do so, you first define a base model, then define one `override` with each alternative grid configuration.

The `scenarios` can combine several `overrides`.
For example, you might also want to explore different future cost developments, and define `overrides` for those.
In your scenarios, you can then combine overrides for a specific realisation of future costs and a specific grid configuration.


Overrides (and the scenarios that reference overrides) can overwrite anything that is defined in the Calliope YAML files: both model configuration and model definition.
