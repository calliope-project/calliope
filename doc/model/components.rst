
================
Model components
================

Index sets
==========

* ``y``: technologies
* ``x``: locations
* ``t``: time steps

The (hard-coded) time step length is 1 hour. However, this 1-hourly resolution can be adjusted over parts of the total length of the dataset by using the :class:`~calliope.TimeSummarizer` class (currently, only support for downsampling is implemented).

Technologies
============

Each node has the following energy balance variables:

* ``s(y, x, t)``: storage level
* ``rs(y, x, t)``: energy resource ⟷ storage (+ production, - consumption)
* ``bs(y, x, t)``: backup resource ⟷ storage (+ production, - consumption)
* ``es(y, x, t)``: storage ⟷ electricity (+ supply, - demand)
* ``os(y, x, t)``: storage ⟷ overflow (+ dissipation, - shortfall)
* ``e(y, x, t)``: node ⟷ grid (+ supply, - demand)

and the following capacity variables:

* ``s_cap(y, x)``: installed storage capacity
* ``r_cap(y, x)``: installed resource ⟷ storage conversion capacity
* ``r_area(y, x)``: installed collector area [m2]
* ``e_cap(y, x)``: installed storage ⟷ electricity conversion capacity

Each technology must define the following parameters, some of which may be zero. By default, all of these except for ``r`` are inherited from the default technology definition (with default values being ``0`` for capacities and ``1`` for efficiencies).

* ``r(y, x, t)``: available energy (+ resource, - demand) [kWh/m2 * hour]
* ``s_cap_max(y)``: max storage size [kWh]
* ``s_loss(y)``: storage loss rate [hour^-1]
* ``s_init(y)``: initial storage [kWh]
* ``r_area_max(y)``: maximum collector area [m2]
* ``r_eff(y)``: conversion efficiency [unitless]
* ``rs_eff(y)``: conversion efficiency (backup primary source) [unitless]
* ``r_cap_max(y)``: maximum conversion [kW]
* ``e_eff(y)``: conversion efficiency [unitless]
* ``e_cap_max(y)``: maximum installed storage ⟷ electricity conversion capacity [kW]

Backup primary energy source
----------------------------

Very basic support is implemented for a backup primary energy source to deliver energy to storage alongside the primary energy source (``r``). Currently this is only used to provide heat to CSP plants during the ``startup_time`` defined in ``model_settings.yaml``.

This makes use of the following additional parameters:

* ``b(y, x, t)``: available energy (backup primary source) [kWh/hour]
* ``b_max(y)``: maximum conversion (backup primary source) [kW]

