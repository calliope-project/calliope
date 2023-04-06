
Objective
#########

min_cost_optimisation
=====================

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
##########

energy_capacity_per_storage_capacity_min
========================================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_storage_cap_min}_\text{node,tech}) \land \neg (\exists (\textit{energy_cap_per_storage_cap_equals}_\text{node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} \geq \textbf{storage_cap}_\text{node,tech} \times \textit{energy_cap_per_storage_cap_min}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_per_storage_capacity_max
========================================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_storage_cap_max}_\text{node,tech}) \land \neg (\exists (\textit{energy_cap_per_storage_cap_equals}_\text{node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} \leq \textbf{storage_cap}_\text{node,tech} \times \textit{energy_cap_per_storage_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_per_storage_capacity_equals
===========================================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_storage_cap_equals}_\text{node,tech}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textbf{storage_cap}_\text{node,tech} \times \textit{energy_cap_per_storage_cap_equals}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_capacity_equals_energy_capacity
========================================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\textit{resource_cap_equals_energy_cap}_\text{node,tech}\mathord{=}\text{true} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{resource_cap}_\text{node,tech} = \textbf{energy_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

force_zero_resource_area
========================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (((\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate})) \land \textit{energy_cap_max}_\text{node,tech}\mathord{=}\text{0})
        \end{array}
        \begin{cases}
            \textbf{resource_area}_\text{node,tech} = 0&\quad
            \\
        \end{cases}

resource_area_per_energy_capacity
=================================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{resource_area}_\text{node,tech} = \textbf{energy_cap}_\text{node,tech} \times \textit{resource_area_per_energy_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_area_capacity_per_loc
==============================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes }
            \\
            \text{if } (\exists (\textit{available_area}_\text{node}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{resource_area}_\text{node,tech}) \leq \textit{available_area}_\text{node}&\quad
            \\
        \end{cases}

energy_capacity_systemwide
==========================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } ((\exists (\textit{energy_cap_equals_systemwide}_\text{tech}) \lor \exists (\textit{energy_cap_max_systemwide}_\text{tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
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
===============================

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
======================================

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
======================================

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
===================================

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
            \text{if } (\text{carrier_tier} \in \text{[in_2]})\land{}(\text{carrier_tier} \in \text{[in_2,in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_2)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_2]})\land{}(\text{carrier_tier} \in \text{[out_2,out_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_3)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_3]})\land{}(\text{carrier_tier} \in \text{[in_2,in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_3)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[in_3]})\land{}(\text{carrier_tier} \in \text{[out_2,out_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_2)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_2]})\land{}(\text{carrier_tier} \in \text{[in_2,in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_2)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_2,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_2]})\land{}(\text{carrier_tier} \in \text{[out_2,out_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=in,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_3)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_3]})\land{}(\text{carrier_tier} \in \text{[in_2,in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_3)}} (\frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier=out_3,carrier,node,tech,timestep} })&\quad
            \text{if } (\text{carrier_tier} \in \text{[out_3]})\land{}(\text{carrier_tier} \in \text{[out_2,out_3]})
            \\
        \end{cases}

conversion_plus_prod_con_to_zero
================================

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
==================

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
======================

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
======================

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
=======================

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
============

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
===========

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
=============================

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
==============

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
            \text{if } (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier))\land{}(\text{run_config.ensure_feasibility}\mathord{=}\text{true})
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) - \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_export}_\text{node,tech,carrier,timestep}) = 0&\quad
            \text{if } (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier))\land{}(\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) + \textbf{unmet_demand}_\text{node,carrier,timestep} + \textbf{unused_supply}_\text{node,carrier,timestep} = 0&\quad
            \text{if } (\neg (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier)))\land{}(\text{run_config.ensure_feasibility}\mathord{=}\text{true})
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{node,tech,carrier,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{node,tech,carrier,timestep}) = 0&\quad
            \text{if } (\neg (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier)))\land{}(\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))
            \\
        \end{cases}

balance_supply
==============

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
======================

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
==============

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
==============================

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
================================

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
=================================

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
===============

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
===================

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
====================

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
======================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\text{tech_group=transmission} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textbf{energy_cap}_\text{node=remote_node,tech=remote_tech}&\quad
            \\
        \end{cases}

export_balance
==============

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
==================

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
====================

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
===========================

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
===========================================

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
============================

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
===========================

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
===========================================

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
===========================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } ((((\text{tech_group=storage} \lor \text{tech_group=supply_plus}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{node,tech} = \textbf{units}_\text{node,tech} \times \textit{storage_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_units_milp
==========================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textit{energy_cap_per_unit}_\text{node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{node,tech} = \textbf{units}_\text{node,tech} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_max_purchase_milp
=================================

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
=================================

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
==================================

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
==================================

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
=============================

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (((\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer}) \land (\exists (\textit{units_max_systemwide}_\text{tech}) \lor \exists (\textit{units_equals_systemwide}_\text{tech}))) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
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
=====================

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
======================

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
==========

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
============

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
#####

cost_var
========

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
===============

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ cost }\negthickspace \in \negthickspace\text{ costs }
            \\
            \text{if } ((\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (0) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_energy_cap}_\text{cost,node,tech}))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus})\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{binary})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{node,tech}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})
            \\
            \textit{annualisation_weight} \times ((\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (0) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech})) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{node,tech}))&\quad
            \text{if } (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}_\text{cost,node,tech}) \land (\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_energy_cap}_\text{cost,node,tech})))\land{}(\neg (\exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \land \text{tech_group=supply_plus}))\land{}(\neg (\exists (\textit{cost_storage_cap}_\text{cost,node,tech}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}_\text{cost,node,tech}) \land (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \lor \textit{cap_method}_\text{node,tech}\mathord{=}\text{integer})))
            \\
        \end{cases}

cost
====

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
            \text{if } ((\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))\land{}(\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}))
            \\
            \textbf{cost_investment}_\text{node,tech,cost}&\quad
            \text{if } ((\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))
            \\
            \sum\limits_{\text{timestep} \in \text{timesteps}} (\textbf{cost_var}_\text{node,tech,cost,timestep})&\quad
            \text{if } (\neg ((\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate})))\land{}(\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep}))
            \\
            0&\quad
            \text{if } (\neg ((\exists (\textit{cost_energy_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual}_\text{cost,node,tech}) \lor \exists (\textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) \lor \exists (\textit{cost_purchase}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_area}_\text{cost,node,tech}) \lor \exists (\textit{cost_resource_cap}_\text{cost,node,tech}) \lor \exists (\textit{cost_storage_cap}_\text{cost,node,tech})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate})))\land{}(\neg (\exists (\textit{cost_export}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_con}_\text{cost,node,tech,timestep}) \lor \exists (\textit{cost_om_prod}_\text{cost,node,tech,timestep})))
            \\
        \end{cases}

Decision Variables
##################

energy_cap
==========

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{energy_cap_min}_\text{node,tech} \leq \textbf{energy_cap}_\text{node,tech}&\quad
            \\
            \textbf{energy_cap}_\text{node,tech} \leq \textit{energy_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_prod
============

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
===========

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
==============

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
=============

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } ((\exists (\textit{resource_area_min}_\text{node,tech}) \lor \exists (\textit{resource_area_max}_\text{node,tech}) \lor \exists (\textit{resource_area_equals}_\text{node,tech}) \lor \exists (\textit{resource_area_per_energy_cap}_\text{node,tech}) \lor \textit{resource_unit}_\text{node,tech}\mathord{=}\text{energy_per_area}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{resource_area_min}_\text{node,tech} \leq \textbf{resource_area}_\text{node,tech}&\quad
            \\
            \textbf{resource_area}_\text{node,tech} \leq \textit{resource_area_max}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_con
============

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
============

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\text{tech_group=supply_plus} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{resource_cap_min}_\text{node,tech} \leq \textbf{resource_cap}_\text{node,tech}&\quad
            \\
            \textbf{resource_cap}_\text{node,tech} \leq \textit{resource_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_cap
===========

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (((\text{tech_group=storage} \lor \text{tech_group=supply_plus}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate})) \land \textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textit{storage_cap_min}_\text{node,tech} \leq \textbf{storage_cap}_\text{node,tech}&\quad
            \\
            \textbf{storage_cap}_\text{node,tech} \leq \textit{storage_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

storage
=======

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
=========

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{binary} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            0 \leq \textbf{purchased}_\text{node,tech}&\quad
            \\
            \textbf{purchased}_\text{node,tech} \leq 1&\quad
            \\
        \end{cases}

units
=====

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\textit{cap_method}_\text{node,tech}\mathord{=}\text{integer} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{units_min}_\text{node,tech} \leq \textbf{units}_\text{node,tech}&\quad
            \\
            \textbf{units}_\text{node,tech} \leq \textit{units_max}_\text{node,tech}&\quad
            \\
        \end{cases}

operating_units
===============

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
===============

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
============

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
=============

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
