
Objective
---------

minmax_cost_optimisation
^^^^^^^^^^^^^^^^^^^^^^^^

Minimise the total cost of installing and operation all technologies in the system. If multiple cost classes are present (e.g., monetary and co2 emissions), the weighted sum of total costs is minimised. Cost class weights can be defined in `run.objective_options.cost_class`.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \min{}
        \end{array}
        \begin{cases}
            \sum\limits_{\text{cost} \in \text{costs}} (\sum\limits_{\substack{\text{node} \in \text{nodes} \\ \text{tech} \in \text{techs}}} (\textbf{cost}_\text{node,tech,cost}) \times \textit{objective_cost_class}_\text{cost}) + \sum\limits_{\text{timestep} \in \text{timesteps}} (\sum\limits_{\substack{\text{carrier} \in \text{carriers} \\ \text{node} \in \text{nodes}}} (\textbf{unmet_demand}_\text{node,carrier,timestep} - \textbf{unused_supply}_\text{node,carrier,timestep}) \times \textit{timestep_weights}_\text{timestep}) \times \textit{bigM}&\quad
            \text{if } (\text{run_config.ensure_feasibility}\mathord{=}\text{true})
            \\
            \sum\limits_{\text{cost} \in \text{costs}} (\sum\limits_{\substack{\text{node} \in \text{nodes} \\ \text{tech} \in \text{techs}}} (\textbf{cost}_\text{node,tech,cost}) \times \textit{objective_cost_class}_\text{cost})&\quad
            \text{if } (\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))
            \\
        \end{cases}

Subject to
----------

energy_capacity_per_storage_capacity_min
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound of a `storage`/`supply_plus` technology's energy capacity relative to its storage capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_storage_cap_min}_\text{node,tech}) \land \neg (\exists (\textit{energy_cap_per_storage_cap_equals}_\text{node,tech})))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} \geq \textbf{storage_cap}_\text{node,tech} \times \textit{energy_cap_per_storage_cap_min}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_per_storage_capacity_max
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound of a `storage`/`supply_plus` technology's energy capacity relative to its storage capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_storage_cap_max}_\text{node,tech}) \land \neg (\exists (\textit{energy_cap_per_storage_cap_equals}_\text{node,tech})))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} \leq \textbf{storage_cap}_\text{node,tech} \times \textit{energy_cap_per_storage_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_per_storage_capacity_equals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a fixed relationship between a `storage`/`supply_plus` technology's energy capacity and its storage capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_storage_cap_equals}_\text{node,tech}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textbf{storage_cap}_\text{node,tech} \times \textit{energy_cap_per_storage_cap_equals}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_capacity_equals_energy_capacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a `supply_plus` technology's energy capacity to equal its resource capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\textit{resource_cap_equals_energy_cap}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{resource_cap}_\text{node,tech} = \textbf{energy_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

force_zero_resource_area
^^^^^^^^^^^^^^^^^^^^^^^^

Set a technology's resource area to zero if its energy capacity upper bound is zero.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } ((\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}) \land \textit{energy_cap_max}_\text{node,tech}\mathord{=}\text{0})
        \end{array}
        \begin{cases}
            \textbf{resource_area}_\text{node,tech} = 0&\quad
            \\
        \end{cases}

resource_area_per_energy_capacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a fixed relationship between a technology's energy capacity and its resource area.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{resource_area_per_energy_cap}_\text{node,tech}))
        \end{array}
        \begin{cases}
            \textbf{resource_area}_\text{node,tech} = \textbf{energy_cap}_\text{node,tech} \times \textit{resource_area_per_energy_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_area_capacity_per_loc
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set an upper bound on the total area that all technologies with a resource_area can occupy at a given node.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes }
            \\
            \text{if } (\exists (\textit{available_area}_\text{node}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{resource_area}_\text{node,tech}) \leq \textit{available_area}_\text{node}&\quad
            \\
        \end{cases}

energy_capacity_systemwide
^^^^^^^^^^^^^^^^^^^^^^^^^^

Set an upper bound on, or a fixed total of, energy capacity of a technology across all nodes in which the technology exists.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_equals_systemwide}_\text{tech}) \lor \exists (\textit{energy_cap_max_systemwide}_\text{tech}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{energy_cap}_\text{node,tech}) = \textit{energy_cap_equals_systemwide}_\text{tech}&\quad
            \text{if } (\exists (\textit{energy_cap_equals_systemwide}_\text{tech}))
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{energy_cap}_\text{node,tech}) \leq \textit{energy_cap_max_systemwide}_\text{tech}&\quad
            \text{if } (\neg (\exists (\textit{energy_cap_equals_systemwide}_\text{tech})))
            \\
        \end{cases}

balance_conversion_plus_primary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fix the relationship between total carrier production and total carrier consumption of `conversion_plus` technologies for `in` (consumption) and `out` (production) carrier flows.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=conversion_plus} \land \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep}\mathord{>}\text{0})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep}) \times \textit{energy_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_max_conversion_plus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound in each timestep of a `conversion_plus` technology's total carrier production on its `out` carrier flows.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=conversion_plus} \land \neg (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) \leq \textit{timestep_resolution}_\text{timestep} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_production_min_conversion_plus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound in each timestep of a `conversion_plus` technology's total carrier production on its `out` carrier flows.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{energy_cap_min_use}_\text{node,tech}) \land \text{tech_group=conversion_plus} \land \neg (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) \geq \textit{timestep_resolution}_\text{timestep} \times \textbf{energy_cap}_\text{node,tech} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

balance_conversion_plus_non_primary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fix the relationship between a `conversion_plus` technology's total `in_2`/`in_3` (consumption) and `out_2`/`out_3` (production) carrier flows and its `in` (consumption) and `out` (production) carrier flows.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier_tier }\negthickspace \in \negthickspace\text{ carrier_tiers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=conversion_plus} \land \text{carrier_tier} \in \text{[in_2,out_2,in_3,out_3]} \land \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep}\mathord{>}\text{0})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_2)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_2,in_3]})\land{}(\text{carrier_tier} \in \text{[in_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_3)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_2,in_3]})\land{}(\text{carrier_tier} \in \text{[in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_2)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_2,in_3]})\land{}(\text{carrier_tier} \in \text{[out_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_3)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_2,in_3]})\land{}(\text{carrier_tier} \in \text{[out_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_2)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_2,out_3]})\land{}(\text{carrier_tier} \in \text{[in_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_3)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_2,out_3]})\land{}(\text{carrier_tier} \in \text{[in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_2)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_2,out_3]})\land{}(\text{carrier_tier} \in \text{[out_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_3)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_2,out_3]})\land{}(\text{carrier_tier} \in \text{[out_3]})
            \\
        \end{cases}

conversion_plus_prod_con_to_zero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a `conversion_plus` technology's carrier flow to zero if its `carrier_ratio` is zero.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep}\mathord{=}\text{0} \land \text{tech_group=conversion_plus})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} = 0&\quad
            \text{if } (\text{carrier_tier} \in \text{[in,in_2,in_3]})
            \\
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} = 0&\quad
            \text{if } (\text{carrier_tier} \in \text{[out,out_2,out_3]})
            \\
        \end{cases}

balance_conversion
^^^^^^^^^^^^^^^^^^

Fix the relationship between a `conversion` technology's carrier production and consumption.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=conversion})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) = -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) \times \textit{energy_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_max
^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound of a non-`conversion_plus` technology's carrier production.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \neg (\text{tech_group=conversion_plus}) \land \neg (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true} \land \text{carrier_tier} \in \text{[out]})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} \leq \textbf{energy_cap}_\text{node,tech} \times \textit{timestep_resolution}_\text{timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_min
^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound of a non-`conversion_plus` technology's carrier production.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \exists (\textit{energy_cap_min_use}_\text{node,tech}) \land \neg (\text{tech_group=conversion_plus}) \land \neg (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true} \land \text{carrier_tier} \in \text{[out]})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} \geq \textbf{energy_cap}_\text{node,tech} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_consumption_max
^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound of a non-`conversion_plus` technology's carrier consumption.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land (\text{tech_group=transmission} \lor \text{tech_group=demand} \lor \text{tech_group=storage}) \land (\neg (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \lor \text{tech_group=demand}) \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true} \land \text{carrier_tier} \in \text{[in]})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \geq -1 \times \textbf{energy_cap}_\text{node,tech} \times \textit{timestep_resolution}_\text{timestep}&\quad
            \\
        \end{cases}

resource_max
^^^^^^^^^^^^

Set the upper bound of a `supply_plus` technology's resource consumption.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            \textbf{resource_con}_\text{node,tech,timestep} \leq \textit{timestep_resolution}_\text{timestep} \times \textbf{resource_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_max
^^^^^^^^^^^

Set the upper bound of the amount of energy a `storage`/`supply_plus` technology can store.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep} - \textbf{storage_cap}_\text{node,tech} \leq 0&\quad
            \\
        \end{cases}

storage_discharge_depth_limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound of the stored energy a `storage`/`supply_plus` technology must keep in reserve at all times.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true} \land \exists (\textit{storage_discharge_depth}_\text{node,tech}))
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep} - (\textit{storage_discharge_depth}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech}) \geq 0&\quad
            \\
        \end{cases}

system_balance
^^^^^^^^^^^^^^

Set the global energy balance of the optimisation problem by fixing the total production of a given energy carrier to equal the total consumption of that carrier at every node in every timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
        \end{array}
        \begin{cases}
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) - \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textbf{unmet_demand}_\text{node,carrier,timestep} + \textbf{unused_supply}_\text{node,carrier,timestep} = 0&\quad
            \text{if } (\text{run_config.ensure_feasibility}\mathord{=}\text{true})\land{}(\sum\limits_{\text{tech} \in \text{techs}} (export_carrier))
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) + \textbf{unmet_demand}_\text{node,carrier,timestep} + \textbf{unused_supply}_\text{node,carrier,timestep} = 0&\quad
            \text{if } (\text{run_config.ensure_feasibility}\mathord{=}\text{true})\land{}(\neg (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier)))
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) - \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) = 0&\quad
            \text{if } (\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))\land{}(\sum\limits_{\text{tech} \in \text{techs}} (export_carrier))
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) = 0&\quad
            \text{if } (\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))\land{}(\neg (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier)))
            \\
        \end{cases}

balance_supply
^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, a `supply` technology's ability to produce energy based on the quantity of  available resource.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{resource}_\text{node,tech,timestep}) \land \text{tech_group=supply})
        \end{array}
        \begin{cases}
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
            \\
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_cap})
            \\
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy})
            \\
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } \leq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}) \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
            \\
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } \leq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}) \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_cap})
            \\
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } \leq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}) \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy})
            \\
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} = 0&\quad
            \text{if } (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
        \end{cases}

balance_supply_min_use
^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound on, or a fixed amount of, the energy a `supply` technology must consume in each timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{resource}_\text{node,tech,timestep}) \land \text{tech_group=supply} \land \exists (\textit{resource_min_use}_\text{node,tech}) \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))
        \end{array}
        \begin{cases}
            \textit{resource_min_use}_\text{node,tech} \leq \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} }&\quad
            \\
        \end{cases}

balance_demand
^^^^^^^^^^^^^^


.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=demand})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
            \\
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy})
            \\
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} \geq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
            \\
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} \geq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} \geq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy})
            \\
        \end{cases}

balance_supply_plus_no_storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, a `supply_plus` (without storage) technology's ability to produce energy based on only the quantity of consumed resource.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=supply_plus} \land \neg (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true}))
        \end{array}
        \begin{cases}
            \textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep} = 0&\quad
            \text{if } (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep} = \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            \text{if } (\neg (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0}))
            \\
        \end{cases}

balance_supply_plus_with_storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, a `supply_plus` (with storage) technology's ability to produce energy based on the quantity of consumed resource and available stored energy.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=supply_plus} \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            \text{if } (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))\land{}(\neg (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0}))
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))\land{}(\neg (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0}))
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}}) \times \textbf{storage}_\text{node,tech,timestep=lookup_cluster_last_timestep} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}}) \times \textbf{storage}_\text{node,tech,timestep=lookup_cluster_last_timestep} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}))\land{}(\neg (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0}))
            \\
        \end{cases}

resource_availability_supply_plus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, a `supply_plus` technology's ability to consume its available energy resource.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{resource}_\text{node,tech,timestep}) \land \text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            \textbf{resource_con}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
            \\
            \textbf{resource_con}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{resource_con}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            \text{if } (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true})\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy})
            \\
            \textbf{resource_con}_\text{node,tech,timestep} \leq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
            \\
            \textbf{resource_con}_\text{node,tech,timestep} \leq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{resource_con}_\text{node,tech,timestep} \leq \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{force_resource}_\text{node,tech}\mathord{=}\text{true}))\land{}(\textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy})
            \\
        \end{cases}

balance_storage
^^^^^^^^^^^^^^^

Fix the quantity of energy stored in a `storage` technology at the end of each timestep based on the net flow of energy charged and discharged and the quantity of energy stored at the start of the timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=storage})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech} - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } - (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech} - (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } - (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} - (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}}) \times \textbf{storage}_\text{node,tech,timestep=lookup_cluster_last_timestep} - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } - (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}}) \times \textbf{storage}_\text{node,tech,timestep=lookup_cluster_last_timestep} - (\textbf{carrier_con}_\text{node,tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
        \end{cases}

set_storage_initial
^^^^^^^^^^^^^^^^^^^

Fix the relationship between energy stored in a `storage` technology at the start and end of the whole model period.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{storage_initial}_\text{node,tech}) \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true} \land \text{run_config.cyclic_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep=timesteps[-1]} \times ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep=timesteps[-1]}}) = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

balance_transmission
^^^^^^^^^^^^^^^^^^^^

Fix the relationship between between energy flowing into and out of a `transmission` link in each timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=transmission} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} = -1 \times \textbf{carrier_con}_\text{node=remote_node,tech=remote_tech,carrier,timestep} \times \textit{energy_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

symmetric_transmission
^^^^^^^^^^^^^^^^^^^^^^

Fix the energy capacity of two `transmission` technologies representing the same link in the system.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\text{tech_group=transmission})
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textbf{energy_cap}_\text{node=remote_node,tech=remote_tech}&\quad
            \\
        \end{cases}

export_balance
^^^^^^^^^^^^^^

Set the lower bound of a technology's carrier production to a technology's carrier export, for any technologies that can export energy out of the system.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \textit{export}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} \geq \textbf{carrier_export}_\text{node,tech,carrier,timestep}&\quad
            \\
        \end{cases}

carrier_export_max
^^^^^^^^^^^^^^^^^^

Set the upper bound of a technology's carrier export, for any technologies that can export energy out of the system.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{export_max}_\text{node,tech}) \land \exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \textit{export}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_export}_\text{node,tech,carrier,timestep} \leq \textit{export_max}_\text{node,tech} \times \textbf{operating_units}_\text{node,tech,timestep}&\quad
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textbf{carrier_export}_\text{node,tech,carrier,timestep} \leq \textit{export_max}_\text{node,tech}&\quad
            \text{if } (\neg (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}))
            \\
        \end{cases}

unit_commitment_milp
^^^^^^^^^^^^^^^^^^^^

Set the upper bound of the number of integer units of technology that can exist, for any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
        \end{array}
        \begin{cases}
            \textbf{operating_units}_\text{node,tech,timestep} \leq \textbf{units}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_production_max_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound of a non-`conversion_plus` technology's ability to produce energy, for any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \neg (\text{tech_group=conversion_plus}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} \leq \textbf{operating_units}_\text{node,tech,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_max_conversion_plus_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound of a `conversion_plus` technology's ability to produce energy across all of its `out` energy carriers, if it uses integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=conversion_plus} \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) \leq \textbf{operating_units}_\text{node,tech,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_consumption_max_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound of a non-`conversion_plus` technology's ability to consume energy, for any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\neg (\text{tech_group=conversion_plus}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \geq -1 \times \textbf{operating_units}_\text{node,tech,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_production_min_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound of a non-`conversion_plus` technology's ability to produce energy, for any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \exists (\textit{energy_cap_min_use}_\text{node,tech}) \land \neg (\text{tech_group=conversion_plus}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} \geq \textbf{operating_units}_\text{node,tech,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_production_min_conversion_plus_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound of a `conversion_plus` technology's ability to produce energy across all of its `out` energy carriers, if it uses integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{energy_cap_min_use}_\text{node,tech}) \land \text{tech_group=conversion_plus} \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) \geq \textbf{operating_units}_\text{node,tech,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_capacity_units_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fix the storage capacity of any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (((\text{tech_group=storage} \lor \text{tech_group=supply_plus}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{node,tech} = \textbf{units}_\text{node,tech} \times \textit{storage_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_units_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^

Fix the energy capacity of any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_unit}_\text{node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textbf{units}_\text{node,tech} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_max_purchase_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, a technology's energy capacity, for any technology with binary capacity purchasing.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\exists (\textit{energy_cap_max}_\text{node,tech}) \lor \exists (\textit{energy_cap_equals}_\text{node,tech})) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textit{energy_cap_equals}_\text{node,tech} \times \textit{energy_cap_scale}_\text{node,tech} \times \textbf{purchased}_\text{node,tech}&\quad
            \text{if } (\exists (\textit{energy_cap_equals}_\text{node,tech}))
            \\
            \textbf{energy_cap}_\text{node,tech} \leq \textit{energy_cap_max}_\text{node,tech} \times \textit{energy_cap_scale}_\text{node,tech} \times \textbf{purchased}_\text{node,tech}&\quad
            \text{if } (\neg (\exists (\textit{energy_cap_equals}_\text{node,tech})))
            \\
        \end{cases}

energy_capacity_min_purchase_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound on a technology's energy capacity, for any technology with binary capacity purchasing.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \exists (\textit{energy_cap_min}_\text{node,tech}) \land \neg (\exists (\textit{energy_cap_equals}_\text{node,tech})) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} \geq \textit{energy_cap_min}_\text{node,tech} \times \textit{energy_cap_scale}_\text{node,tech} \times \textbf{purchased}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_capacity_max_purchase_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, a technology's storage capacity, for any technology with binary capacity purchasing.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\exists (\textit{storage_cap_max}_\text{node,tech}) \lor \exists (\textit{storage_cap_equals}_\text{node,tech})) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{node,tech} = \textit{storage_cap_equals}_\text{node,tech} \times \textbf{purchased}_\text{node,tech}&\quad
            \text{if } (\exists (\textit{storage_cap_equals}_\text{node,tech}))
            \\
            \textbf{storage_cap}_\text{node,tech} \leq \textit{storage_cap_max}_\text{node,tech} \times \textbf{purchased}_\text{node,tech}&\quad
            \text{if } (\neg (\exists (\textit{storage_cap_equals}_\text{node,tech})))
            \\
        \end{cases}

storage_capacity_min_purchase_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the lower bound on a technology's storage capacity, for any technology with binary capacity purchasing.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \exists (\textit{storage_cap_min}_\text{node,tech}) \land \neg (\exists (\textit{storage_cap_equals}_\text{node,tech})) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{node,tech} \geq \textit{storage_cap_min}_\text{node,tech} \times \textbf{purchased}_\text{node,tech}&\quad
            \\
        \end{cases}

unit_capacity_systemwide_milp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the upper bound on, or a fixed total of, the total number of units of a technology that can be purchased across all nodes where the technology can exist, for any technology using integer units to define its capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } ((\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \land (\exists (\textit{units_max_systemwide}_\text{tech}) \lor \exists (\textit{units_equals_systemwide}_\text{tech})))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{purchased}_\text{node,tech}) = \textit{units_equals_systemwide}_\text{tech}&\quad
            \text{if } (\exists (\textit{units_equals_systemwide}_\text{tech}))\land{}(\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{units}_\text{node,tech}) = \textit{units_equals_systemwide}_\text{tech}&\quad
            \text{if } (\exists (\textit{units_equals_systemwide}_\text{tech}))\land{}(\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{purchased}_\text{node,tech}) \leq \textit{units_max_systemwide}_\text{tech}&\quad
            \text{if } (\neg (\exists (\textit{units_equals_systemwide}_\text{tech})))\land{}(\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{units}_\text{node,tech}) \leq \textit{units_max_systemwide}_\text{tech}&\quad
            \text{if } (\neg (\exists (\textit{units_equals_systemwide}_\text{tech})))\land{}(\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
        \end{cases}

asynchronous_con_milp
^^^^^^^^^^^^^^^^^^^^^

Set a technology's ability to consume energy in the same timestep that it is producing energy, for any technology using the asynchronous production/consumption binary switch.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{force_asynchronous_prod_con}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) \leq (1 - \textbf{prod_con_switch}_\text{node,tech,timestep}) \times \textit{bigM}&\quad
            \\
        \end{cases}

asynchronous_prod_milp
^^^^^^^^^^^^^^^^^^^^^^

Set a technology's ability to produce energy in the same timestep that it is consuming energy, for any technology using the asynchronous production/consumption binary switch.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{force_asynchronous_prod_con}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) \leq \textbf{prod_con_switch}_\text{node,tech,timestep} \times \textit{bigM}&\quad
            \\
        \end{cases}

ramping_up
^^^^^^^^^^

Set the upper bound on a technology's ability to ramp energy production up beyond a certain percentage compared to the previous timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{energy_ramping}_\text{node,tech,timestep}) \land \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]}))
        \end{array}
        \begin{cases}
            \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} } \leq \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true}))
            \\
            \frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} } \leq \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true}))
            \\
            \frac{ (\textbf{carrier_con}_\text{node,tech,carrier,timestep} + \textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ (\textbf{carrier_con}_\text{node,tech,carrier,timestep-1} + \textbf{carrier_prod}_\text{node,tech,carrier,timestep-1}) }{ \textit{timestep_resolution}_\text{timestep-1} } \leq \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{node,tech}&\quad
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
            \\
        \end{cases}

ramping_down
^^^^^^^^^^^^

Set the upper bound on a technology's ability to ramp energy production down beyond a certain percentage compared to the previous timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{energy_ramping}_\text{node,tech,timestep}) \land \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]}))
        \end{array}
        \begin{cases}
            -1 \times \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{node,tech} \leq \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }&\quad
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true}))
            \\
            -1 \times \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{node,tech} \leq \frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }&\quad
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true}))
            \\
            -1 \times \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{node,tech} \leq \frac{ (\textbf{carrier_con}_\text{node,tech,carrier,timestep} + \textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ (\textbf{carrier_con}_\text{node,tech,carrier,timestep-1} + \textbf{carrier_prod}_\text{node,tech,carrier,timestep-1}) }{ \textit{timestep_resolution}_\text{timestep-1} }&\quad
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true} \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true})
            \\
        \end{cases}

Where
-----

cost_var
^^^^^^^^

The operating costs per timestep of a technology

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ cost }\negthickspace \in \negthickspace\text{ costs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}))
        \end{array}
        \begin{cases}
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \text{carrier_tier} \in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier} \in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\neg (\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \text{carrier_tier} \in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier} \in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \text{carrier_tier} \in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier} \in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\neg (\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \text{carrier_tier} \in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier} \in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{node,tech,timestep})&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \text{carrier_tier} \in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier} \in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \exists (\textit{cost_export}_\text{cost,node,tech,timestep}))\land{}(\neg (\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{node,tech,timestep})&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=supply} \land \textit{energy_eff}_\text{node,tech,timestep}\mathord{>}\text{0} \land \text{carrier_tier} \in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}))&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier} \in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (0)&\quad
            \text{if } (\neg (\exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep})))\land{}(\neg (\exists (\textit{cost_om_con}_\text{cost,node,tech,timestep})))
            \\
        \end{cases}

cost_investment
^^^^^^^^^^^^^^^

The installation costs of a technology, including annualised investment costs and annual maintenance costs.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ cost }\negthickspace \in \negthickspace\text{ costs }
            \\
            \text{if } (\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech}))
        \end{array}
        \begin{cases}
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (0) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (0) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))
            \\
        \end{cases}

cost
^^^^

The total annualised costs of a technology, including installation and operation costs.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ cost }\negthickspace \in \negthickspace\text{ costs }
            \\
            \text{if } (\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}))
        \end{array}
        \begin{cases}
            \textbf{cost_investment}_\text{node,tech,cost} + \sum\limits_{\text{timestep} \in \text{timesteps}} (\textbf{cost_var}_\text{node,tech,cost,timestep})&\quad
            \text{if } (\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}))
            \\
            \textbf{cost_investment}_\text{node,tech,cost}&\quad
            \text{if } (\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))
            \\
            \sum\limits_{\text{timestep} \in \text{timesteps}} (\textbf{cost_var}_\text{node,tech,cost,timestep})&\quad
            \text{if } (\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}))
            \\
            0&\quad
            \text{if } (\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))
            \\
        \end{cases}

Decision Variables
------------------

energy_cap
^^^^^^^^^^

A technology's energy capacity, also known as its nominal or nameplate capacity.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } None
        \end{array}
        \begin{cases}
            \textit{energy_cap_min}_\text{node,tech} \leq \textbf{energy_cap}_\text{node,tech}&\quad
            \\
            \textbf{energy_cap}_\text{node,tech} \leq \textit{energy_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_prod
^^^^^^^^^^^^

The energy produced by a technology per timestep, also known as the energy discharged (from `storage` technologies) or the energy received (by `transmission` technologies) on a link.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_prod}_\text{node,tech}\mathord{=}\text{true} \land \text{carrier_tier} \in \text{[out,out_2,out_3]})
        \end{array}
        \begin{cases}
            0 \leq \textbf{carrier_prod}_\text{node,tech,carrier,timestep}&\quad
            \\
            \textbf{carrier_prod}_\text{node,tech,carrier,timestep} \leq inf&\quad
            \\
        \end{cases}

carrier_con
^^^^^^^^^^^

The energy consumed by a technology per timestep, also known as the energy consumed (by `storage` technologies) or the energy sent (by `transmission` technologies) on a link.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{carrier}_\text{carrier_tier,carrier,tech}) \land \textit{allowed_carrier_con}_\text{node,tech}\mathord{=}\text{true} \land \text{carrier_tier} \in \text{[in,in_2,in_3]})
        \end{array}
        \begin{cases}
            -inf \leq \textbf{carrier_con}_\text{node,tech,carrier,timestep}&\quad
            \\
            \textbf{carrier_con}_\text{node,tech,carrier,timestep} \leq 0&\quad
            \\
        \end{cases}

carrier_export
^^^^^^^^^^^^^^

The energy exported outside the system boundaries by a technology per timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textit{export_carrier}_\text{carrier,node,tech}) \land \textit{export}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0 \leq \textbf{carrier_export}_\text{node,tech,carrier,timestep}&\quad
            \\
            \textbf{carrier_export}_\text{node,tech,carrier,timestep} \leq inf&\quad
            \\
        \end{cases}

resource_area
^^^^^^^^^^^^^

The area in space utilised directly (e.g., solar PV panels) or indirectly (e.g., biofuel crops) by a technology.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})
        \end{array}
        \begin{cases}
            \textit{resource_area_min}_\text{node,tech} \leq \textbf{resource_area}_\text{node,tech}&\quad
            \\
            \textbf{resource_area}_\text{node,tech} \leq \textit{resource_area_max}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_con
^^^^^^^^^^^^

The energy consumed from outside the system boundaries by a `supply_plus` technology.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            0 \leq \textbf{resource_con}_\text{node,tech,timestep}&\quad
            \\
            \textbf{resource_con}_\text{node,tech,timestep} \leq inf&\quad
            \\
        \end{cases}

resource_cap
^^^^^^^^^^^^

The upper limit on energy that can be consumed from outside the system boundaries by a `supply_plus` technology in each timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            \textit{resource_cap_min}_\text{node,tech} \leq \textbf{resource_cap}_\text{node,tech}&\quad
            \\
            \textbf{resource_cap}_\text{node,tech} \leq \textit{resource_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_cap
^^^^^^^^^^^

The upper limit on energy that can be stored by a `supply_plus` or `storage` technology in any timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } ((\text{tech_group=storage} \lor \text{tech_group=supply_plus}) \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textit{storage_cap_min}_\text{node,tech} \leq \textbf{storage_cap}_\text{node,tech}&\quad
            \\
            \textbf{storage_cap}_\text{node,tech} \leq \textit{storage_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

storage
^^^^^^^

The energy stored by a `supply_plus` or `storage` technology in each timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } ((\text{tech_group=storage} \lor \text{tech_group=supply_plus}) \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0 \leq \textbf{storage}_\text{node,tech,timestep}&\quad
            \\
            \textbf{storage}_\text{node,tech,timestep} \leq inf&\quad
            \\
        \end{cases}

purchased
^^^^^^^^^

Binary switch defining whether a technology has been purchased or not, for any technology set to require binary capacity purchasing. This is used to set a fixed cost for a technology, irrespective of its installed capacity, on top of which a cost for the quantity of installed capacity can also be applied.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            0 \leq \textbf{purchased}_\text{node,tech}&\quad
            \\
            \textbf{purchased}_\text{node,tech} \leq 1&\quad
            \\
        \end{cases}

units
^^^^^

Integer number of a technology that has been purchased, for any technology set to require inteter capacity purchasing. This is used to allow installation of fixed capacity units of technologies. Since technology capacity is no longer a continuous decision variable, it is possible for these technologies to have a lower bound set on carrier production/consumption which will only be enforced in those timesteps that the technology is operating (otherwise, the same lower bound forces the technology to produce/consume that minimum amount of energy in *every* timestep).

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
        \end{array}
        \begin{cases}
            \textit{units_min}_\text{node,tech} \leq \textbf{units}_\text{node,tech}&\quad
            \\
            \textbf{units}_\text{node,tech} \leq \textit{units_max}_\text{node,tech}&\quad
            \\
        \end{cases}

operating_units
^^^^^^^^^^^^^^^

Integer number of a technology that is operating in each timestep, for any technology set to require inteter capacity purchasing.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
        \end{array}
        \begin{cases}
            0 \leq \textbf{operating_units}_\text{node,tech,timestep}&\quad
            \\
            \textbf{operating_units}_\text{node,tech,timestep} \leq inf&\quad
            \\
        \end{cases}

prod_con_switch
^^^^^^^^^^^^^^^

Binary switch to force asynchronous carrier production/consumption of a `storage`/`supply_plus`/`transmission` technology. This ensures that a technology with carrier flow efficiencies < 100% cannot produce and consume energy simultaneously to remove unwanted energy from the system.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\textit{force_asynchronous_prod_con}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0 \leq \textbf{prod_con_switch}_\text{node,tech,timestep}&\quad
            \\
            \textbf{prod_con_switch}_\text{node,tech,timestep} \leq 1&\quad
            \\
        \end{cases}

unmet_demand
^^^^^^^^^^^^

Virtual source of energy to ensure model feasibility. This should only be considered a debugging rather than a modelling tool as it may distort the model in other ways due to the large impact it has on the objective function value. When present in a model in which it has been requested, it indicates an inability for technologies in the model to reach a sufficient combined supply capacity to meet demand.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{run_config.ensure_feasibility}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0 \leq \textbf{unmet_demand}_\text{node,carrier,timestep}&\quad
            \\
            \textbf{unmet_demand}_\text{node,carrier,timestep} \leq inf&\quad
            \\
        \end{cases}

unused_supply
^^^^^^^^^^^^^

Virtual sink of energy to ensure model feasibility. This should only be considered a debugging rather than a modelling tool as it may distort the model in other ways due to the large impact it has on the objective function value. In model results, the negation of this variable is combined with `unmet_demand` and presented as only one variable: `unmet_demand`. When present in a model in which it has been requested, it indicates an inability for technologies in the model to reach a sufficient combined consumption capacity to meet required energy production (e.g. from renewables without the possibility of curtailment).

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{run_config.ensure_feasibility}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -inf \leq \textbf{unused_supply}_\text{node,carrier,timestep}&\quad
            \\
            \textbf{unused_supply}_\text{node,carrier,timestep} \leq 0&\quad
            \\
        \end{cases}
