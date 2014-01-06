
===========
Constraints
===========

-----------------
Basic constraints
-----------------

Objective function
------------------

Provided by: :func:`calliope.constraints.base.model_objective`

The objective function minimizes cost:

.. math::

   min: z = \sum_y (weight(y) \times \sum_x cost(y, x, k))

where :math:`k=monetary`.

Node energy balance
-------------------

Provided by: :func:`calliope.constraints.base.node_energy_balance`

``e`` is constrained by three equations:

.. math::

   e(c, y, x, t) = e_{prod}(c, y, x, t) + e_{con}(c, y, x, t)

   e_{prod}(c, y, x, t) = e_{s,prod}(c, y, x, t) \times e_{eff}(y, x, t)

   e_{con}(c, y, x, t) = e_{s,con}(c, y, x, t) \times eff

where :math:`eff = 1` if ``c`` is the ``source_carrier``, and otherwise, either :math:`1 / e_{eff}(y, x, t)` or 0 if :math:`e_{eff}(y, x, t)` is 0.

``rs`` is constrained depending on the context:

* If :math:`r(y, x, t)` is was set to infinity, no constraint is applied to ``r``.
* If a technology ``y`` is not allowed at a location ``x``, or if :math:`r(y, x, t) = 0`, then :math:`r_{s}(y, x, y) = 0`.

Finally, in all other cases, two stages apply: first, :math:`r_{avail}` is determined as follows:

.. math::

   r_{avail} = r(y, x, t) * r_{scale}(y) * r_{area}(y) * r_{eff}(y)

Then, one of the following three constraints are applied:

* :math:`r_{s}(y, x, t) = r_{avail}` if ``force_r(y)`` is true.
* :math:`r_{s}(y, x, t) \leq r_{avail}` if ``force_r(y)`` if :math:`r_{s}(y, x, t) > 0`.
* :math:`r_{s}(y, x, t) = r_{avail}` if ``force_r(y)`` otherwise.

Finally, the centerpiece of node energy balancing is the storage balance equation. There are two general cases:

* Transmission technologies
* All other technologies

For the case of transmission technologies, special logic is applied to link the current ``y`` and ``x`` with the required ``y_remote`` and ``x_remote``:

.. math::

   es_{prod}(carrier, y, x, t) = -1 \times es_{con}(carrier, y_remote, x_remote, t) / e_{eff}(y, x, t)

In other word, storage is bypassed entirely.

For all other technologies, the following equation applies:

.. math::

   s(y, x, t) &= s_{minusone} + r_{s}(y, x, t) + r_{secs}(y, x, t) \\
   &- \sum_c es_{prod}(c, y, x, t) \\
   &- \sum_c es_{con}(c, y, x, t) - o_{s}(y, x, t)

with :math:`s_{minusone}` defined as follows:

* If no storage is allowed at this node (i.e. if :math:`s_{cap,max}(y) = 0`) then :math:`s_{minusone} = 0`.
* Else, if the current ``t`` is not the first ``t``, then the following equation applies:

   .. math::

      s_{minus one} = 1 - s_{loss}(y)^{timeres(t-1)} * s(y, x, t-1)

   :math:`timeres(t)` returns the time step resolution of the given :math:`t`.
* Finally, if the current ``t`` is the first ``t``, :math:`s_{minus one}` is initialized with either the value given in ``s_init(y, x)`` or in operation mode, with ``s_init`` carried over from a previous optimization period.

Node build constraints
----------------------

Provided by: :func:`calliope.constraints.base.node_constraints_build`

Built capacity is constrained by four equations:

.. math::

   s_{cap}(y, x) \leq s_{cap,max}(y, x)

   r_{cap}(y, x) \leq r_{cap,max}(y, x)

   r_{area}(y, x) \leq r_{area,max}(y, x)

   e_{cap}(y, x) \leq e_{cap,max}(y, x)

If a technology ``y`` is not allowed at a location ``x``, :math:`e_{cap}(y, x) = 0` is used.

In the case of both ``r_cap`` and ``e_cap``, if the respective ``cap_max`` option has been set to infinite, no constraint at all is set up rather than the equations given above.

In operation mode, the :math:`\leq` turns into an equality, so that the first equation reads :math:`s_{cap}(y, x) = s_{cap,max}(y, x)`, and the others are modified analogously.

Node operational constraints
----------------------------

Provided by: :func:`calliope.constraints.base.node_constraints_operational`

The variable :math:`r_{s}(y, x, t)` is constrained to remain within :math:`r_{cap}(y, x)`:

.. math::

   r_{s}(y, x, t) \leq timeres(t) \times (r_{cap}(y, x) / r_{eff}(y))

   r_{s}(y, x, t) \geq -1 \times timeres(t) \times (r_{cap}(y, x) / r_{eff}(y))

:math:`e_{s}(c, y, x, t)` is constrained by

.. math::

   e_{s,prod}(c, y, x, y) \leq timeres(t) \times (e_{cap}(y, x) / e_{eff,ref}(y, x))

if ``c`` is the ``carrier`` of ``y``, else :math:`e_{s,prod}(c, y, x, y) = 0`.

If ``e_cap_min_use`` is defined, the minimum output is constrained by

.. math::

   e_{s,prod}(c, y, x, y) \geq timeres(t) \times (e_{cap}(y, x) / e_{eff,ref}(y, x)) \times e_{cap,minuse}

Analogous to the above, if ``c`` is the ``carrier`` of ``y``, and if ``e_can_be_negative`` is true, then

.. math::

   e_{s,con}(c, y, x, y) \geq -1 \times timeres(t) \times (e_{cap}(y, x) / e_{eff,ref}(y, x))

and :math:`e_{s,con}(c, y, x, y) = 0` otherwise. There is however an additional special case, for transmission technologies there ``c`` is the ``source_carrier`` of ``y``, where the following equation replaces the above one:

.. math::

   e_{s,con}(x, y, x, t) = -1 \times e_{s,prod}(carrier, y, x, t)

where :math:`carrier` is the (primary) carrier of technology ``y``.

Storage cannot exceed its maximum size:

.. math::

   s(y, x, t) \leq s_{cap}(y, x)

And finally, the secondary resource (:math:`r_{sec,s}`) is allowed during the hours within ``startup_time`` and only if the technology allows this:

.. math::

   r_{sec,s}(y, x, t) = timeres(t) \times e_{cap}(y, x) / e_{eff}(y, x)

Otherwise, it is :math:`r_{sec,s}(y, x, t) = 0`.

Transmission constraints
------------------------

Provided by: :func:`calliope.constraints.base.transmission_constraints`

These force :math:`e_{cap}` to be symmetric for transmission nodes, for a given transmission line between :math:`x_1` and :math:`x_2`:

.. math::

   e_{cap}(y_1, x_1) = e_{cap}(y_1, x_2)

Node costs
----------

Provided by: :func:`calliope.constraints.base.node_costs`

These equations compute costs per node.

The depreciation rate for each cost class ``k`` is calculated as

.. math::

   d(y, k) = 1 / plant\_life(y)

if the interest rate :math:`i` is 0, else

.. math::

   d(y, k) = \frac{i \times (1 + i(y, k))^{plant\_life(k)}}{(1 + i(y, k))^{plant\_life(k)} - 1}

Costs are split up into construction and operation costs:

.. math::

   cost(y, x, k) = cost_{con}(y, x, k) + cost_{op}(y, x, k)

   cost_{con}(y, x, k) &= d(y, k) \times \frac{\sum_t timeres(t)}{8760} \\
   & \times (cost_{s\_cap}(y, k) * s_{cap}(y, x) \\
   & + cost_{r\_cap}(y, k) * r_{cap}(y, x) \\
   & + cost_{r\_area}(y, k) * r_{area}(y, x) \\
   & + cost_{e\_cap}(y, k) * e_{cap}(y, x))

   cost_{op}(y, x, k) &= cost_{om\_frac}(y, k) \times cost_{con}(y, x, k) \\
   & + cost_{om\_var}(y, k) \times \sum_t e_{prod}(c, y, x, t) \\
   & + cost_{om\_fuel}(y, k) \times \sum_t r_{s}(y, x, t)


Model constraints
-----------------

Provided by: :func:`calliope.constraints.base.model_constraints`

Model-wide balancing constraints are constructed for nodes that have children. They differentiate between three cases:

* ``c = power``
* ``c = heat``
* All other ``c``

In the first case, a balancing equation applies:

.. math::

   \sum_y \sum_{xs} e_{prod}(c, y, xs, t) = 0 \qquad\text{for each } t

Where :math:`xs` are all the :math:`x` in a family, determined by taking the parent and all child nodes at lower levels.

This equality is a :math:`\geq` inequality in the second case.

In the final case, no balancing constraint is applied at all.

.. _optional_constraints:

--------------------
Optional constraints
--------------------

Ramping
-------

Provided by: :func:`calliope.constraints.ramping.ramping_rate`

Constraints the rate at which plants can adjust their output, for technologies that define ``constraints.e_ramping``:

.. math::

   diff = e(c, y, x, t) - e(c, y, x, t-1)

   maxrate = e_{ramping} \times timeres(t) \times e_{cap}(y, x)

   diff \leq maxrate

   diff \geq -1 \times max_ramping_rate

----------------------------
Loading optional constraints
----------------------------

Additional constraints can be loaded by specifying two options in ``model.yaml``:

* ``constraints_pre_load:`` Will be evaluated just before loading constraints. Any Python code can be given here, for example ``import`` statements to import custom constraints.
* ``constraints:`` A list of constraints to load in addition to the default constraints, e.g. ``['constraints.ramping.ramping_rate']``

For example, the following settings would load two custom constraints from ``my_custom_module``::

   constraints_pre_load: 'import my_custom_module'
   contraints: ['my_custom_module.my_constraint0',
                'my_custom_module.my_constraint1']

Custom constraints have access to all model configuration (see :doc:`configuration`) and any number of additional configuration directives can be set on a per-technology, per-location or model-wide basis for custom constraints.
