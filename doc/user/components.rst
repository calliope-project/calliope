
================
Model components
================

The following details the basic approach to modeling the energy system used in Calliope, as well as the main components and terminology.

-----------
Terminology
-----------

The terminology defined here is used throughout the documentation and the model code and configuration files:

* **Technology**: a technology that produces, consumes, converts or transports energy
* **Location**: a site which can contain multiple technologies and which may contain other locations for energy balancing purposes
* **Node**: a combination of technology and location resulting in specific energy balance equations (:ref:`see below <node_energy_balance>`)
* **Resource**: a source or sink of energy that can (or must) be used by a technology to introduce into or remove energy from the system
* **Carrier**: an energy carrier that groups technologies together into the same network, for example ``electricity`` or ``heat``.

As more generally in constrained optimization, the following terms are also used:

* Parameter: a fixed coefficient that enters into model equations
* Variable: a variable coefficient (decision variable) that enters into model equations
* Set: an index in the algebraic formulation of the equations
* Constraint: an equality or inequality expression that constrains one or several variables

----------
Index sets
----------

Most parameters, variables, and constraints are formulated with respect to at least some of the indices below:

* ``c``: carriers
* ``y``: technologies
* ``x``: locations
* ``t``: time steps
* ``k``: cost classes

In some cases, these index sets may have only a single member. For example, if only the power system is modeled, the set ``c`` (carriers) will have a single member, ``power``.

----------------
Technology types
----------------

Each technology (that is, each member of the set ``y``) is of a specific *technology type*, which determines how the framework models the technology and what properties it can have. The technology type is specified by inheritance from one of five abstract base technologies (see the :doc:`model configuration section <configuration>` for more details on this inheritance model):

* Supply: Supplies energy from a resource to a carrier (a source). Can have storage. Can define an additional secondary resource (base technology: ``supply``)
* Demand: Acts like supply but with a resource that is negative (a sink). Draws energy from a carrier to satisfy a resource demand. Can also have storage (base technology: ``demand``)
* Conversion: Converts energy from one carrier to another, can have neither resource nor storage associated with it (base technology: ``conversion``)
* Storage: Can store energy of a specific carrier, cannot have any resource (base technology: ``storage``)
* Transmission: Transports energy of a specific carrier from one location to another, can have neither resource nor storage (base technology: ``transmission``)

------------
Cost classes
------------

Costs are modeled in Calliope via *cost classes*. By default, only two classes are defined, ``monetary`` and ``emissions``.

Technologies can define costs for components (installed capacity) and for operation & maintenance, for any cost class.

The primary cost class, ``monetary``, is used to calculate levelized costs and by default enters into the objective function. Therefore each technology should define at least one type of ``monetary`` cost, as it would be considered free otherwise.

By default, any cost not specified is assumed to be zero.

The ``emissions`` cost class is not entered into the objective function but used to account for greenhouse gas emissions.

Additional cost classes can be created simply by adding them to the definition of costs for a technology (see the :doc:`model configuration section <configuration>` for more detail on this). For example, emissions could be broken up into different classes, like ``co2`` and ``nox``.

--------------------------------------------------
Putting technologies and locations together: Nodes
--------------------------------------------------

In the model definition, locations can be defined, and for each location (or for groups of locations), technologies can be permitted. The details of this are laid out in the :doc:`model configuration section <configuration>`.

A *node* is the combination of a specific location and technology, and is how Calliope internally builds the model. For a given location, ``x``, and technology, ``y``, a set of equations defined over ``(x, y)`` models that specific node.

The most important node variables are laid out below, but more detail is also available in the section :doc:`formulation`.

.. _node_energy_balance:

-------------------
Node energy balance
-------------------

The basic formulation of each node uses an energy balance to make sure that

Each node has the following energy balance variables:

* ``s(y, x, t)``: storage level at time ``t``
* ``rs(y, x, t)``: resource to/from storage (+ production, - consumption) at time ``t``
* ``rbs(y, x, t)``: secondary resource to storage (+ production) at time ``t``
* ``es(c, y, x, t)``: storage to/from carrier in default case (+ supply, - demand) at time ``t``
* ``ec(c, y, x, t)``: conversion to/from carrier in case with parasitics (+ supply, - demand) at time ``t``

For most technologies, ``ec`` is not actually defined, and ``es`` directly converts storage to carrier. ``ec`` is used for technologies where a difference between gross and net installed conversion capacity must be made (technologies which specify an internal energy use).

.. TODO add a figure with the basic layout of the node and different variables going through from resource to storage to carrier (with the possible extra step of - es - ec - e)

Internally, ``e``, ``es`` and ``ec`` are split into separate variables, for the positive and negative components, i.e. ``e_prod`` and ``e_con`` (analogously for ``es`` and ``ec``). This simplifies the formulation of some constraints. In the documentation, unless necessary in a specific context, the combined (e.g. ``e``) notation is used for simplicity.

The secondary resource can deliver energy to storage via ``rbs`` alongside the primary energy source (via ``rs``), but only if the necessary setting (``constraints.allow_rsec:``) is enabled for a technology. Optionally, this can be allowed only during the ``startup_time:`` (defined in the model-wide settings), e.g. to allow storage to be filled up initially.

Each node also has the following capacity variables:

* ``s_cap(y, x)``: installed storage capacity
* ``r_cap(y, x)``: installed resource to storage conversion capacity
* ``r_area(y, x)``: installed collector area
* ``e_cap(y, x)``: installed storage to carrier conversion capacity
* ``rb_cap(y, x)``: installed secondary resource to storage conversion capacity

For nodes that have an internal (parasitic) energy consumption, ``e_cap_net(y, x)`` specifies the net storage capacity while ``e_cap(y, x)`` is gross capacity. If no internal energy consumption is specified, ``e_cap(y, x)`` is the net (and gross) capacity. ``e_cap_net`` is always calculated by the model and cannot be set or constrained manually.

When defining a technology, it must be given at least some constraints, that is, options that describe the functioning of the technology. If not specified, all of these are inherited from the default technology definition (with default values being ``0`` for capacities and ``1`` for efficiencies). Some examples of such options are:

* ``r(y, x, t)``: available resource (+ source, - sink)
* ``s_cap_max(y)``: maximum storage capacity
* ``s_loss(y)``: storage loss rate
* ``r_area_max(y)``: maximum resource collector area
* ``r_eff(y)``: resource conversion efficiency
* ``r_cap_max(y)``: maximum resource to storage conversion capacity
* ``e_eff(y)``: maximum storage to carrier conversion efficiency
* ``e_cap_max(y)``: maximum installed storage to/from carrier conversion capacity

.. Note:: Generally, these constraints are defined on a per-technology basis. However, some (but not all) of them may be overridden on a per-location basis. This allows, for example, setting different constraints on the allowed maximum capacity for a specific technology at each location separately. See :doc:`configuration` for details on this.

Finally, each node tracks its costs, split in three basic parts:

* ``cost_con``: construction costs
* ``cost_op_fixed``: fixed operational and maintenance (O&M) costs (i.e., per installed capacity)
* ``cost_op_var``: variable O&M costs (i.e., per produced output)

The next section, :doc:`formulation`, details the constraints that actually implement all these formulations mathematically. The section following it, :doc:`configuration`, details how a model is configured, and how the various components outlined here are defined in a working model.
