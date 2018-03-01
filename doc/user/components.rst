
===========================
Components to build a model
===========================

This section proves an overview of how a model is built using Calliope.

Calliope allows a modeler to define technologies with arbitrary characteristics by "inheriting" basic traits from a number of included base technologies, :ref:`which are described below <technology_types>`. Technologies can take a **resource** from outside of the modeled system and turn it into a specific energy **carrier** in the system. These technologies, together with the **locations** specified in the model, result in a set of **nodes**: the energy balance equations indexed over the set of technologies and locations.


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

* ``carriers``: carriers
* ``techs``: technologies
* ``locs``: locations
* ``timesteps``: time steps
* ``costs``: cost classes

In some cases, these index sets may have only a single member. For example, if only the power system is modeled, the set ``carriers`` will have a single member, ``power``.

When processed, these sets are often concatenated to avoid sparse matrices. For instance, if a technology ``boiler`` only exists in location ``X1`` and not in locations ``X2`` or ``X3``, then we will specify parameters for just the ``loc::tech`` ``X1::boiler``. This can be extended to parameters which also consider ``carriers``, such that we would have a ``loc::tech::carrier`` ``X1::boiler::heat`` (avoiding empty parameter values for ``power``, as the boiler never considers that enery carrier).

.. _technology_types:

----------------
Technology types
----------------

Each technology (that is, each member of the set ``techs``) is of a specific *technology type*, which determines how the framework models the technology and what properties it can have. The technology type is specified by inheritance from one of seven abstract base technologies (see :ref:`configuration_techs` in the model configuration section for more details on this inheritance model):

* Supply: Supplies energy from a resource to a carrier (a source) (base technology: ``supply``)
* Supply_plus: A more feature rich version of ``supply``. It can have storage of resource before conversion to carrier, can define an additional secondary resource, and can have several more intermediate loss factors (base technology: ``supply_plus``)
* Demand: Acts like supply but with a resource that is negative (a sink). Draws energy from a carrier to satisfy a resource demand (base technology: ``demand``)
* Conversion: Converts energy from one carrier to another, can have neither resource nor storage associated with it (base technology: ``conversion``)
* Conversion_plus: A more feature rich version of ``conversion``. There can be several carriers in, converted to several carriers out (base technology: ``conversion_plus``)
* Storage: Can store energy of a specific carrier, cannot have any resource (base technology: ``storage``)
* Transmission: Transports energy of a specific carrier from one location to another, can have neither resource nor storage (base technology: ``transmission``)

The internal definition of these abstract base technologies is given in the :ref:`configuration reference <abstract_base_tech_definitions>`.

------------
Cost classes
------------

Costs are modeled in Calliope via *cost classes*. By default, only one classes is defined: ``monetary``.

Technologies can define costs for components (installed capacity), for operation & maintenance, and for export for any cost class. Costs can be given as negative values, which defines a revenue rather than a cost.

The primary cost class, ``monetary``, is used to calculate levelized costs and by default enters into the objective function. Therefore each technology should define at least one cost parameter, as it would be considered free otherwise. By default, any cost not specified is assumed to be zero.

Only the ``monetary`` cost class is entered into the default objective function, but other cost classes can be defined for accounting purposes, e.g. ``emissions`` to account for greenhouse gas emissions. Additional cost classes can be created simply by adding them to the definition of costs for a technology (see the :doc:`model configuration section <configuration>` for more detail on this).

To add additional cost classes to the objective function (e.g. ``emissions``), a custom objective function would need to be created. See :ref:`config_reference_model_wide` in model configuration for more details.

Revenue
-------

It is possible to specify revenues for technologies simply by setting a negative cost value. For example, to consider a feed-in tariff for PV generation, it could be given a negative operational cost equal to the real operational cost minus the level of feed-in tariff received.

--------------------------------------------------
Putting technologies and locations together: Nodes
--------------------------------------------------

In the model definition, locations can be defined, and for each location (or for groups of locations), technologies can be permitted. The details of this are laid out in the :doc:`model configuration section <configuration>`.

A *node* is the combination of a specific location and technology, and is how Calliope internally builds the model. For a given location, ``loc``, and technology, ``tech``, a set of equations defined over ``loc::tech`` models that specific node.

The most important node variables are laid out below, but more detail is also available in the section :doc:`formulation`.

.. _node_energy_balance:

-------------------
Node energy balance
-------------------

The basic formulation of each node uses a set of energy balance equations. Depending on the technology type, different energy balance variables are used:

* ``storage(loc::tech, timestep)``: storage level at time ``timestep``
    This is used for ``storage`` and ``supply_plus`` technologies.
* ``resource(loc::tech, timestep)``: resource to technology (+ production) at time ``timestep``. If storage is defined for ``supply_plus``, this is resource to storage flow.
    This is used for ``supply_plus`` technologies.
* ``carrier_prod(loc::tech::carrier, timestep)``: production of a given energy carrier by a technology (+ supply) at time ``timestep``.
    This is used for all technologies, except ``demand``.
* ``c_con(loc::tech::carrier, timestep)``: consumption of a given energy carrier by a technology at time ``timestep``
    This is used for all technologies, except ``supply`` and ``supply_plus``.

The resulting losses associated with energy balancing also depend on the technology type. Each technology node is mapped here, with details on interactions given in :doc:`configuration`.

.. figure:: images/nodes.*
   :alt: Layout of a various node and their energy balance

   The layout of nodes, and their energy balance variables, associated with each technology type. The outward arrows show where losses occur. Depending on a technology, some of these steps may be skipped. For example, most ``supply_plus`` technologies will have no parasitic losses.

Each node can also have the following capacity variables:

* ``storage_cap(loc::tech)``: installed storage capacity
    This is used for ``storage`` and ``supply_plus`` technologies.
* ``resource_cap(loc::tech)``: installed resource to storage conversion capacity
    This is used for ``supply_plus`` technologies.
* ``resource_area(loc::tech)``: installed resource collector area
    This is used for ``supply``, ``supply_plus``, and ``demand`` technologies.
* ``energy_cap(loc::tech)``: installed storage to carrier conversion capacity
    This is used for all technologies.

.. Note:: For nodes that have an internal (parasitic) energy consumption, ``energy_cap_net`` is also included in the solution. This specifies the net conversion capacity, while ``energy_cap`` is gross capacity.

When defining a technology, it must be given at least some constraints, that is, options that describe the functioning of the technology. If not specified, all of these are inherited from the default technology definition (with default values being ``0`` for capacities and ``1`` for efficiencies). Some examples of such options are:

* ``resource(loc::tech, timestep)``: available resource (+ source, - sink)
* ``storage_cap_max(loc::tech)``: maximum storage capacity
* ``storage_loss(loc::tech, timestep)``: storage loss rate
* ``resource_area_max(loc::tech)``: maximum resource collector area
* ``resource_eff(loc::tech)``: resource efficiency
* ``resource_cap_max(loc::tech)``: maximum resource to storage conversion capacity
* ``energy_eff(loc::tech, timestep)``: resource/storage/carrier_in to carrier_out conversion efficiency
* ``energy_cap_max(loc::tech)``: maximum installed carrier conversion capacity, applied to carrier_out

.. Note:: Generally, these constraints are defined on a per-technology basis. However, some (but not all) of them may be overridden on a per-location basis. This allows, for example, setting different constraints on the allowed maximum capacity for a specific technology at each location separately. See :doc:`configuration` for details on this. Once processed in Calliope, all constraints will be indexed over location::technology sets.

Finally, each node tracks its costs (+ costs, - revenue), formulated in two constraints (more details in the :doc:`formulation` section):

* ``cost_investment``: static investment costs, for construction and fixed operational and maintenance (O&M) (i.e., costs per unit of installed capacity)
* ``cost_var``: variable O&M and export costs (i.e., costs per produced unit of output)

.. Note:: Efficiencies, available resources, and costs can be defined to vary in time. Equally (and more likely) they can be given as single values. For more detail on time-varying versus constant values, see :ref:`the corresponding section <time_varying_vs_constant_parameters>` in the model formulation chapter.

-------------------
Linking locations
-------------------
Locations are linked together by transmission technologies. By consuming an energy carrier in one location and outputting it in another, linked location, transmission technologies allow resources to be drawn from the system at a different location from where they are brought into it.

.. figure:: images/nodes_network.*
   :alt: Layout of linked locations

   Schematic of location linking, including interaction of resource, nodes, and energy carriers. The dashed box defines the system under consideration. Resource flows (green) are lossless, whereas losses can occur along transmission links (black).

Transmission links are considered by the system as nodes at each end of the link, with the same technology at each end. In this regard, the same nodal energy balance equations apply. Additionally, the user can utilise per-distance constraints and costs. For more information on available constraints/costs, see the :doc:`configuration` section.

The next section is a brief tutorial. Following this, :doc:`formulation` details the constraints that actually implement all these formulations mathematically. The section following it, :doc:`configuration`, details how a model is configured, and how the various components outlined here are defined in a working model.
