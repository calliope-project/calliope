
================
Model components
================

-----------
Terminology
-----------

.. TODO

* **Technology**: a technology that produces, consumes, converts or transports energy
* **Location**: a site which can contain multiple technologies and which may contain other locations for energy balancing purposes
* **Node**: a combination of technology and location resulting in specific energy balance equations (:ref:`see below <node_energy_balance>`)
* **Resource**: a source or sink of energy that can (or must) be used by a technology to introduce into or remove energy from the system
* **Carrier**: an energy carrier that groups technologies together into the same network, for example ``electricity`` or ``heat``.

----------
Index sets
----------

* ``c``: carriers
* ``y``: technologies
* ``x``: locations
* ``t``: time steps
* ``k``: cost classes

.. _node_energy_balance:

-------------------
Node energy balance
-------------------

Each node, that is, combination of location and technology, has the following energy balance variables:

* ``s(y, x, t)``: storage level
* ``rs(y, x, t)``: resource to/from storage (+ production, - consumption)
* ``rsecs(y, x, t)``: secondary resource to/from storage (+ production, - consumption)
* ``os(y, x, t)``: storage to/from overflow (+ dissipation, - shortfall)
* ``e(c, y, x, t)``: carrier to/from grid (+ supply, - demand)
* ``es(c, y, x, t)``: storage to/from carrier (+ supply, - demand)

Internally, ``e`` and ``es`` are split into two variables, for the positive and negative components.

It also defines the following capacity variables:

* ``s_cap(y, x)``: installed storage capacity
* ``r_cap(y, x)``: installed resource to storage conversion capacity
* ``r_area(y, x)``: installed collector area [m2]
* ``e_cap(y, x)``: installed storage to electricity conversion capacity

Internally, the variables ``es_prod(y, x, t)`` and ``es_con(y, x, t)`` are also used for flux from storage to electricity (always > 0), and flux from electricity to storage (always < 0) respectively. This is used to allow formulation of certain linear constraints.

Each technology must define the following parameters, some of which may be zero. By default, all of these are inherited from the default technology definition (with default values being ``0`` for capacities and ``1`` for efficiencies). Some of these may be overridden on a per-location basis (see :doc:`configuration`).

* ``r(y, x, t)``: available energy (+ resource, - demand) [kWh/m2 * hour]
* ``s_cap_max(y)``: max storage size [kWh]
* ``s_loss(y)``: storage loss rate [hour^-1]
* ``s_init(y)``: initial storage [kWh]
* ``r_area_max(y)``: maximum collector area [m2]
* ``r_eff(y)``: conversion efficiency [unitless]
* ``r_cap_max(y)``: maximum conversion [kW]
* ``e_eff(y)``: conversion efficiency [unitless]
* ``e_cap_max(y)``: maximum installed storage ⟷ electricity conversion capacity [kW]

Secondary resource
==================

Basic support is implemented for a secondary resource (``rsec``) to deliver energy to storage alongside the primary energy source (``r``). This is only allowed during the ``startup_time`` defined in the model settings, for technologies that set ``constraints.allow_rsec`` to ``true``. The secondary resource is infinite and is constrained by the installed ``e_cap``.

------------
Cost classes
------------

The primary cost class is ``monetary``, which is used to calculate levelized costs and by default enters into the objective function. Therefore each technology should define at least one type of ``monetary`` cost.

The ``emissions`` cost class allows emissions accounting.

Other cost classes can easily be added to account for additional positive or negative effects of technologies or e.g. break up emissions into more detailed components.

----------------
Technology types
----------------

.. TODO

* Supply: Supplies energy from a resource to a carrier; can have storage; can define an additional secondary resource
* Demand: Acts like supply but with a resource that is negative. Draws energy from a carrier to satisfy a resource; can have storage
* Conversion: Converts energy from one carrier to another
* Storage: Can store energy of a specific carrier; has no resource
* Transport: Transports energy of a specific carrier from one location to another; has no storage
