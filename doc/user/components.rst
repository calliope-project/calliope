
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

Each technology (that is, each member of the set ``y``) is of a specific *technology type*, which determines how the framework models the technology and what properties it can have. The technology type is specified by inheritance from one of seven abstract base technologies (see :ref:`configuration_techs` in the model configuration section for more details on this inheritance model):

* Supply: Supplies energy from a resource to a carrier (a source) (base technology: ``supply``)
* Supply_plus: A more feature_rich version of ``supply``. It can have storage of resource before conversion to carrier, can define an additional secondary resource, and can have several more intermediate loss factors (base technology: ``supply_plus``)
* Demand: Acts like supply but with a resource that is negative (a sink). Draws energy from a carrier to satisfy a resource demand (base technology: ``demand``)
* Conversion: Converts energy from one carrier to another, can have neither resource nor storage associated with it (base technology: ``conversion``)
* Conversion_plus: A more feature rich version of ``conversion``. There can be several carriers in, converted to several carriers out (base technology: ``conversion_plus``)
* Storage: Can store energy of a specific carrier, cannot have any resource (base technology: ``storage``)
* Transmission: Transports energy of a specific carrier from one location to another, can have neither resource nor storage (base technology: ``transmission``)

------------
Cost classes
------------

Costs are modeled in Calliope via *cost classes*. By default, only one classes is defined: ``monetary``.

Technologies can define costs for components (installed capacity), for operation & maintenance, and for export for any cost class. costs can be given as negative, to define a revenue source.

The primary cost class, ``monetary``, is used to calculate levelized costs and by default enters into the objective function. Therefore each technology should define at least one type of ``monetary`` cost, as it would be considered free otherwise. By default, any cost not specified is assumed to be zero.

Only the ``monetary`` cost class is entered into the default objective function, but other cost classes can be defined for accounting purposes, e.g. ``emissions`` to account for greenhouse gas emissions. Additional cost classes can be created simply by adding them to the definition of costs for a technology (see the :doc:`model configuration section <configuration>` for more detail on this).

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

The basic formulation of each node uses a set of energy balance equations. Depending on the technology type, different energy balance variables are used:

* ``s(y, x, t)``: storage level at time ``t``
    This is used for ``storage`` and ``supply_plus`` technologies.
* ``r(y, x, t)``: resource to technology (+ production) at time ``t``. If storage is defined for ``supply_plus``, this is resource to storage flow.
    This is used for ``supply_plus`` technologies.
* ``r2(y, x, t)``: secondary resource to technology at time ``t``
    This is used for ``supply_plus`` technologies.
* ``c_prod(c, y, x, t)``: production of a given energy carrier by a technology (+ supply) at time ``t``.
    This is used for all technologies, except ``demand``.
* ``c_con(c, y, x, t)``: consumption of a given energy carrier by a technology at time ``t``
    This is used for all technologies, except ``supply`` and ``supply_plus``.

The resulting losses associated with energy balancing also depend on the technology type. Each technology node is mapped here, with details on interactions given in :doc:`configuration`.

.. figure:: images/nodes.*
   :alt: Layout of a various node and their energy balance

   The layout of nodes, and their energy balance variables, associated with each technology type. The outward arrows show where losses occur. Depending on a technology, some of these steps may be skipped. For example, most ``supply_plus`` technologies will have no parasitic losses.

The secondary resource can deliver energy to storage via ``r_2`` alongside the primary energy source (via ``r``), but only if the necessary setting (``constraints.allow_r2:``) is enabled for a technology. Optionally, this can be allowed only during the ``startup_time:`` (defined in the model-wide settings), e.g. to allow storage to be filled up initially.

Each node can also have the following capacity variables:

* ``s_cap(y, x)``: installed storage capacity
    This is used for ``storage`` and ``supply_plus`` technologies.
* ``r_cap(y, x)``: installed resource to storage conversion capacity
    This is used for ``supply_plus`` technologies.
* ``r_area(y, x)``: installed resource collector area
    This is used for ``supply``, ``supply_plus``, and ``demand`` technologies.
* ``e_cap(y, x)``: installed storage to carrier conversion capacity
    This is used for all technologies,.
* ``r2_cap(y, x)``: installed secondary resource to storage conversion capacity
    This is used for ``supply_plus`` technologies.

.. Note:: For nodes that have an internal (parasitic) energy consumption, ``e_cap_net`` is also included in the solution. This specifies the net conversion capacity, while ``e_cap(y, x)`` is gross capacity.

When defining a technology, it must be given at least some constraints, that is, options that describe the functioning of the technology. If not specified, all of these are inherited from the default technology definition (with default values being ``0`` for capacities and ``1`` for efficiencies). Some examples of such options are:

* ``resource(y, x, t)``: available resource (+ source, - sink)
* ``s_cap.max(y)``: maximum storage capacity
* ``s_loss(y, t)``: storage loss rate
* ``r_area.max(y)``: maximum resource collector area
* ``r_eff(y)``: resource conversion efficiency
* ``r_cap.max(y)``: maximum resource to storage conversion capacity
* ``e_eff(y, t)``: maximum storage to carrier conversion efficiency
* ``e_cap.max(y)``: maximum installed storage to/from carrier conversion capacity

.. Note:: Generally, these constraints are defined on a per-technology basis. However, some (but not all) of them may be overridden on a per-location basis. This allows, for example, setting different constraints on the allowed maximum capacity for a specific technology at each location separately. See :doc:`configuration` for details on this.

Finally, each node tracks its costs (+ costs, - revenue), formulated in two constraints (more details in the :doc:`formulation` section):

* ``cost_fixed``: construction and fixed operational and maintenance (O&M) costs
* ``cost_op_var``: variable O&M and export costs (i.e., per produced output)

.. Note:: Efficiencies, available resource, and costs can be defined to vary in time. Equally (and more likely) they can be given as single values.

The next section is a brief tutorial. Following this, :doc:`formulation` details the constraints that actually implement all these formulations mathematically. The section following it, :doc:`configuration`, details how a model is configured, and how the various components outlined here are defined in a working model.
