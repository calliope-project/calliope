# Basic concepts

## What Calliope does

TBA

## How the world is represented in Calliope

A Calliope model is a collection of interconnected technologies, nodes and carriers describing a real world system of flows.
FIXME: Calliope implements the math [LINK TO MATH DOCS] for all of these, allowing you, the user, to concentrate on describing your real-world system.
Usually, we consider those to be _energy_ flows (or in the case of a power system model, _electricity_ flows), and most of what you will read in this documentation concerns energy systems.
However, the concepts are just as applicable to other types of flows, such as water.

The most important concepts are:

**Carriers** are commodities whose flows we track, e.g., electricity, heat, hydrogen, water, CO<sub>2</sub>.

**Technologies** supply, consume, convert, store or transmit _carriers_, e.g., transmission lines/pipes, batteries, power plants, wind turbines, or home appliances.

**Nodes** contain groups of _technologies_ and are usually geographic, e.g., a country, municipality or a single house.

Flows can enter the system from **sources**, e.g., energy from the sun to power a solar panel, and can exit it into **sinks**, e.g., electricity consumed by household appliances.
Unlike _carriers_, we do not explicitly track the type of commodity described by sources and sinks.

Putting all of these possibilities together allows a modeller to create a model that is as simple or complex as necessary to answer a given research question.
Calliope's syntax ensures these models are intuitive, and easy to understand and share.

![Visual description of the Calliope terminology.](../img/description_of_system.svg)

!!! example
    Refer to the [examples and tutorials section](../examples/index.md) for a more practical look at how to build a Calliope model.

## Building blocks of models in Calliope

### YAML: keys and values

TBA

### Maths and data

TBA

### Overrides and scenarios

TBA
