
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
            \sum\limits_{\text{cost} \in \text{costs}} (\sum\limits_{\substack{\text{node} \in \text{nodes} \\ \text{tech} \in \text{techs}}} (\textbf{cost}_\text{tech,node,cost}) \times \textit{objective_cost_class}_\text{cost}) + \sum\limits_{\text{timestep} \in \text{timesteps}} (\sum\limits_{\substack{\text{carrier} \in \text{carriers} \\ \text{node} \in \text{nodes}}} (\textbf{unmet_demand}_\text{carrier,node,timestep} - \textbf{unused_supply}_\text{carrier,node,timestep}) \times \textit{timestep_weights}_\text{timestep}) \times \textit{bigM}&\quad
            if (\text{run_config.ensure_feasibility}\mathord{=}\text{true})
            \\
            \sum\limits_{\text{cost} \in \text{costs}} (\sum\limits_{\substack{\text{node} \in \text{nodes} \\ \text{tech} \in \text{techs}}} (\textbf{cost}_\text{tech,node,cost}) \times \textit{objective_cost_class}_\text{cost})&\quad
            if (\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))
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
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{energy_cap_per_storage_cap_min}) \land \neg (\exists (\textit{energy_cap_per_storage_cap_equals})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node}\geq{}\textbf{storage_cap}_\text{tech,node} \times \textit{energy_cap_per_storage_cap_min}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_per_storage_capacity_max
========================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{energy_cap_per_storage_cap_max}) \land \neg (\exists (\textit{energy_cap_per_storage_cap_equals})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node}\leq{}\textbf{storage_cap}_\text{tech,node} \times \textit{energy_cap_per_storage_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_per_storage_capacity_equals
===========================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{energy_cap_per_storage_cap_equals}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node} = \textbf{storage_cap}_\text{tech,node} \times \textit{energy_cap_per_storage_cap_equals}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_capacity_equals_energy_capacity
========================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\textit{resource_cap_equals_energy_cap}\mathord{=}\text{true} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{resource_cap}_\text{tech,node} = \textbf{energy_cap}_\text{tech,node}&\quad
            \\
        \end{cases}

force_zero_resource_area
========================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}) \land \textit{energy_cap_max}\mathord{=}\text{0})
        \end{array}
        \begin{cases}
            \textbf{resource_area}_\text{tech,node} = 0&\quad
            \\
        \end{cases}

resource_area_per_energy_capacity
=================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{resource_area_per_energy_cap}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{resource_area}_\text{tech,node} = \textbf{energy_cap}_\text{tech,node} \times \textit{resource_area_per_energy_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_area_capacity_per_loc
==============================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes
            \\
            if (\exists (\textit{available_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{resource_area}_\text{tech,node})\leq{}\textit{available_area}_\text{node}&\quad
            \\
        \end{cases}

energy_capacity_systemwide
==========================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            tech \in techs
            \\
            if (\exists (\textit{energy_cap_equals_systemwide}) \lor \exists (\textit{energy_cap_max_systemwide}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{energy_cap}_\text{tech,node}) = \textit{energy_cap_equals_systemwide}_\text{tech}&\quad
            if (\exists (\textit{energy_cap_equals_systemwide}))
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{energy_cap}_\text{tech,node})\leq{}\textit{energy_cap_max_systemwide}_\text{tech}&\quad
            if (\neg (\exists (\textit{energy_cap_equals_systemwide})))
            \\
        \end{cases}

balance_conversion_plus_primary
===============================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=conversion_plus} \land \textit{carrier_ratios}\mathord{>}\text{0})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep}) \times \textit{energy_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_max_conversion_plus
======================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=conversion_plus} \land \neg (\textit{cap_method}\mathord{=}\text{integer}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep})\leq{}\textit{timestep_resolution}_\text{timestep} \times \textbf{energy_cap}_\text{tech,node}&\quad
            \\
        \end{cases}

carrier_production_min_conversion_plus
======================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\exists (\textit{energy_cap_min_use}) \land \text{tech_group=conversion_plus} \land \neg (\textit{cap_method}\mathord{=}\text{integer}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep})\geq{}\textit{timestep_resolution}_\text{timestep} \times \textbf{energy_cap}_\text{tech,node} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

balance_conversion_plus_non_primary
===================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier_tier \in carrier_tiers, 
            timestep \in timesteps
            \\
            if (\text{tech_group=conversion_plus} \land \text{carrier_tier}\in \text{[in_2,out_2,in_3,out_3]} \land \textit{carrier_ratios}\mathord{>}\text{0})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_2)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[in_2,in_3]})\land{}(\text{carrier_tier}\in \text{[in_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_3)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[in_2,in_3]})\land{}(\text{carrier_tier}\in \text{[in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_2)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[in_2,in_3]})\land{}(\text{carrier_tier}\in \text{[out_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_3)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[in_2,in_3]})\land{}(\text{carrier_tier}\in \text{[out_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_2)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[out_2,out_3]})\land{}(\text{carrier_tier}\in \text{[in_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(in_3)}} (\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[out_2,out_3]})\land{}(\text{carrier_tier}\in \text{[in_3]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_2)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[out_2,out_3]})\land{}(\text{carrier_tier}\in \text{[out_2]})
            \\
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} }) = \sum\limits_{\text{carrier} \in \text{carrier_tier(out_3)}} (\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{carrier_ratios}_\text{carrier_tier,carrier,node,tech,timestep} })&\quad
            if (\text{carrier_tier}\in \text{[out_2,out_3]})\land{}(\text{carrier_tier}\in \text{[out_3]})
            \\
        \end{cases}

conversion_plus_prod_con_to_zero
================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\textit{carrier_ratios}\mathord{=}\text{0} \land \text{tech_group=conversion_plus})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} = 0&\quad
            if (\text{carrier_tier}\in \text{[in,in_2,in_3]})
            \\
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep} = 0&\quad
            if (\text{carrier_tier}\in \text{[out,out_2,out_3]})
            \\
        \end{cases}

balance_conversion
==================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=conversion})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) = -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}) \times \textit{energy_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_max
======================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land \neg (\text{tech_group=conversion_plus}) \land \neg (\textit{cap_method}\mathord{=}\text{integer}) \land \textit{allowed_carrier_prod}\mathord{=}\text{true} \land \text{carrier_tier}\in \text{[out]})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep}\leq{}\textbf{energy_cap}_\text{tech,node} \times \textit{timestep_resolution}_\text{timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_min
======================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land \exists (\textit{energy_cap_min_use}) \land \neg (\text{tech_group=conversion_plus}) \land \neg (\textit{cap_method}\mathord{=}\text{integer}) \land \textit{allowed_carrier_prod}\mathord{=}\text{true} \land \text{carrier_tier}\in \text{[out]})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep}\geq{}\textbf{energy_cap}_\text{tech,node} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_consumption_max
=======================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land (\text{tech_group=transmission} \lor \text{tech_group=demand} \lor \text{tech_group=storage}) \land (\neg (\textit{cap_method}\mathord{=}\text{integer}) \lor \text{tech_group=demand}) \land \textit{allowed_carrier_con}\mathord{=}\text{true} \land \text{carrier_tier}\in \text{[in]})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{carrier,tech,node,timestep}\geq{}-1 \times \textbf{energy_cap}_\text{tech,node} \times \textit{timestep_resolution}_\text{timestep}&\quad
            \\
        \end{cases}

resource_max
============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            \textbf{resource_con}_\text{tech,node,timestep}\leq{}\textit{timestep_resolution}_\text{timestep} \times \textbf{resource_cap}_\text{tech,node}&\quad
            \\
        \end{cases}

storage_max
===========

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{include_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{tech,node,timestep} - \textbf{storage_cap}_\text{tech,node}\leq{}0&\quad
            \\
        \end{cases}

storage_discharge_depth_limit
=============================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{include_storage}\mathord{=}\text{true} \land \exists (\textit{storage_discharge_depth}))
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{tech,node,timestep} - (\textit{storage_discharge_depth}_\text{node,tech} \times \textbf{storage_cap}_\text{tech,node})\geq{}0&\quad
            \\
        \end{cases}

system_balance
==============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
        \end{array}
        \begin{cases}
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}) - \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textbf{unmet_demand}_\text{carrier,node,timestep} + \textbf{unused_supply}_\text{carrier,node,timestep} = 0&\quad
            if (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier))\land{}(\text{run_config.ensure_feasibility}\mathord{=}\text{true})
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}) - \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) = 0&\quad
            if (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier))\land{}(\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}) + \textbf{unmet_demand}_\text{carrier,node,timestep} + \textbf{unused_supply}_\text{carrier,node,timestep} = 0&\quad
            if (\neg (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier)))\land{}(\text{run_config.ensure_feasibility}\mathord{=}\text{true})
            \\
            \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \sum\limits_{\text{tech} \in \text{techs}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}) = 0&\quad
            if (\neg (\sum\limits_{\text{tech} \in \text{techs}} (export_carrier)))\land{}(\neg (\text{run_config.ensure_feasibility}\mathord{=}\text{true}))
            \\
        \end{cases}

balance_supply
==============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{resource}) \land \text{tech_group=supply})
        \end{array}
        \begin{cases}
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{tech,node}&\quad
            if (\textit{force_resource}\mathord{=}\text{true} \land \textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_area})
            \\
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\textit{force_resource}\mathord{=}\text{true} \land \textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_cap})
            \\
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            if (\textit{force_resource}\mathord{=}\text{true} \land \textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{resource_unit}\mathord{=}\text{energy})
            \\
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} }\leq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{tech,node}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}) \land \textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_area})
            \\
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} }\leq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}) \land \textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_cap})
            \\
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} }\leq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}) \land \textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{resource_unit}\mathord{=}\text{energy})
            \\
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep} = 0&\quad
            if (\textit{energy_eff}\mathord{=}\text{0})
            \\
        \end{cases}

balance_supply_min_use
======================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{resource}) \land \text{tech_group=supply} \land \exists (\textit{resource_min_use}) \land \textit{energy_eff}\mathord{>}\text{0} \land \neg (\textit{force_resource}\mathord{=}\text{true}))
        \end{array}
        \begin{cases}
            \textit{resource_min_use}_\text{node,tech}\leq{}\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} }&\quad
            \\
        \end{cases}

balance_demand
==============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{tech_group=demand})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{tech,node}&\quad
            if (\textit{force_resource}\mathord{=}\text{true})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_area})
            \\
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\textit{force_resource}\mathord{=}\text{true})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            if (\textit{force_resource}\mathord{=}\text{true})\land{}(\textit{resource_unit}\mathord{=}\text{energy})
            \\
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep}\geq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{tech,node}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}))\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_area})
            \\
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep}\geq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}))\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep}\geq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}))\land{}(\textit{resource_unit}\mathord{=}\text{energy})
            \\
        \end{cases}

balance_supply_plus_no_storage
==============================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{tech_group=supply_plus} \land \neg (\textit{include_storage}\mathord{=}\text{true}))
        \end{array}
        \begin{cases}
            \textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep} = 0&\quad
            if (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0})
            \\
            \textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep} = \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            if (\neg (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0}))
            \\
        \end{cases}

balance_supply_plus_with_storage
================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{tech_group=supply_plus} \land \textit{include_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{tech,node,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{tech,node} + (\textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0})\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep-1}} \times \textbf{storage}_\text{tech,node,timestep-1} + (\textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0})\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true} \lor \neg (\textit{timesteps}\mathord{=}\text{timesteps[0]}) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}} \times \textbf{storage}_\text{tech,node,timestep=lookup_cluster_last_timestep} + (\textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0})\land{}(\exists (\textit{lookup_cluster_last_timestep}))
            \\
            \textbf{storage}_\text{tech,node,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{tech,node} + (\textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            if (\neg (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0}))\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep-1}} \times \textbf{storage}_\text{tech,node,timestep-1} + (\textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            if (\neg (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0}))\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true} \lor \neg (\textit{timesteps}\mathord{=}\text{timesteps[0]}) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}} \times \textbf{storage}_\text{tech,node,timestep=lookup_cluster_last_timestep} + (\textbf{resource_con}_\text{tech,node,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            if (\neg (\textit{energy_eff}\mathord{=}\text{0} \lor \textit{parasitic_eff}\mathord{=}\text{0}))\land{}(\exists (\textit{lookup_cluster_last_timestep}))
            \\
        \end{cases}

resource_availability_supply_plus
=================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\exists (\textit{resource}) \land \text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            \textbf{resource_con}_\text{tech,node,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{tech,node}&\quad
            if (\textit{force_resource}\mathord{=}\text{true})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_area})
            \\
            \textbf{resource_con}_\text{tech,node,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\textit{force_resource}\mathord{=}\text{true})\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{resource_con}_\text{tech,node,timestep} = \textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            if (\textit{force_resource}\mathord{=}\text{true})\land{}(\textit{resource_unit}\mathord{=}\text{energy})
            \\
            \textbf{resource_con}_\text{tech,node,timestep}\leq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{resource_area}_\text{tech,node}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}))\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_area})
            \\
            \textbf{resource_con}_\text{tech,node,timestep}\leq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}))\land{}(\textit{resource_unit}\mathord{=}\text{energy_per_cap})
            \\
            \textbf{resource_con}_\text{tech,node,timestep}\leq{}\textit{resource}_\text{node,tech,timestep} \times \textit{resource_scale}_\text{node,tech}&\quad
            if (\neg (\textit{force_resource}\mathord{=}\text{true}))\land{}(\textit{resource_unit}\mathord{=}\text{energy})
            \\
        \end{cases}

balance_storage
===============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{tech_group=storage})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{tech,node,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{tech,node} - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } - (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep-1}} \times \textbf{storage}_\text{tech,node,timestep-1} - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } - (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{>}\text{0})\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true} \lor \neg (\textit{timesteps}\mathord{=}\text{timesteps[0]}) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}} \times \textbf{storage}_\text{tech,node,timestep=lookup_cluster_last_timestep} - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{energy_eff}_\text{node,tech,timestep} } - (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{>}\text{0})\land{}(\exists (\textit{lookup_cluster_last_timestep}))
            \\
            \textbf{storage}_\text{tech,node,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{tech,node} - (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{=}\text{0})\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep-1}} \times \textbf{storage}_\text{tech,node,timestep-1} - (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{=}\text{0})\land{}(\textit{timesteps}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true} \lor \neg (\textit{timesteps}\mathord{=}\text{timesteps[0]}) \land \neg (\exists (\textit{lookup_cluster_last_timestep})))
            \\
            \textbf{storage}_\text{tech,node,timestep} = 1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep=lookup_cluster_last_timestep}} \times \textbf{storage}_\text{tech,node,timestep=lookup_cluster_last_timestep} - (\textbf{carrier_con}_\text{carrier,tech,node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep})&\quad
            if (\textit{energy_eff}\mathord{=}\text{0})\land{}(\exists (\textit{lookup_cluster_last_timestep}))
            \\
        \end{cases}

set_storage_initial
===================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{storage_initial}) \land \textit{include_storage}\mathord{=}\text{true} \land \text{run_config.cyclic_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{tech,node,timestep=timesteps[-1]} \times (1 - \textit{storage_loss}_\text{node,tech,timestep}^{\textit{timestep_resolution}_\text{timestep=timesteps[-1]}}) = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{tech,node}&\quad
            \\
        \end{cases}

balance_transmission
====================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{tech_group=transmission} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep} = -1 \times \textbf{carrier_con}_\text{carrier,tech=remote\_tech,node=remote\_node,timestep} \times \textit{energy_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

symmetric_transmission
======================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\text{tech_group=transmission} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node} = \textbf{energy_cap}_\text{tech=remote\_tech,node=remote\_node}&\quad
            \\
        \end{cases}

export_balance
==============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{export_carrier}) \land \textit{export}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep}\geq{}\textbf{carrier_export}_\text{carrier,tech,node,timestep}&\quad
            \\
        \end{cases}

carrier_export_max
==================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{export_max}) \land \exists (\textit{export_carrier}) \land \textit{export}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_export}_\text{carrier,tech,node,timestep}\leq{}\textit{export_max}_\text{node,tech} \times \textbf{operating_units}_\text{tech,node,timestep}&\quad
            if (\textit{cap_method}\mathord{=}\text{integer})
            \\
            \textbf{carrier_export}_\text{carrier,tech,node,timestep}\leq{}\textit{export_max}_\text{node,tech}&\quad
            if (\neg (\textit{cap_method}\mathord{=}\text{integer}))
            \\
        \end{cases}

unit_commitment_milp
====================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{cap_method}\mathord{=}\text{integer})
        \end{array}
        \begin{cases}
            \textbf{operating_units}_\text{tech,node,timestep}\leq{}\textbf{units}_\text{tech,node}&\quad
            \\
        \end{cases}

carrier_production_max_milp
===========================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land \neg (\text{tech_group=conversion_plus}) \land \textit{cap_method}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep}\leq{}\textbf{operating_units}_\text{tech,node,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech,timestep}&\quad
            \\
        \end{cases}

carrier_production_max_conversion_plus_milp
===========================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=conversion_plus} \land \textit{cap_method}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep})\leq{}\textbf{operating_units}_\text{tech,node,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_consumption_max_milp
============================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\neg (\text{tech_group=conversion_plus}) \land \textit{cap_method}\mathord{=}\text{integer} \land \textit{allowed_carrier_con}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_con}_\text{carrier,tech,node,timestep}\geq{}-1 \times \textbf{operating_units}_\text{tech,node,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_production_min_milp
===========================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land \exists (\textit{energy_cap_min_use}) \land \neg (\text{tech_group=conversion_plus}) \land \textit{cap_method}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep}\geq{}\textbf{operating_units}_\text{tech,node,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_production_min_conversion_plus_milp
===========================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\exists (\textit{energy_cap_min_use}) \land \text{tech_group=conversion_plus} \land \textit{cap_method}\mathord{=}\text{integer} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep})\geq{}\textbf{operating_units}_\text{tech,node,timestep} \times \textit{timestep_resolution}_\text{timestep} \times \textit{energy_cap_per_unit}_\text{node,tech} \times \textit{energy_cap_min_use}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_capacity_units_milp
===========================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\text{tech_group=storage} \lor \text{tech_group=supply_plus} \land \textit{cap_method}\mathord{=}\text{integer} \land \textit{include_storage}\mathord{=}\text{true} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{tech,node} = \textbf{units}_\text{tech,node} \times \textit{storage_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_units_milp
==========================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{energy_cap_per_unit}) \land \textit{cap_method}\mathord{=}\text{integer} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node} = \textbf{units}_\text{tech,node} \times \textit{energy_cap_per_unit}_\text{node,tech}&\quad
            \\
        \end{cases}

energy_capacity_max_purchase_milp
=================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{cost_purchase}) \land (\exists (\textit{energy_cap_max}) \lor \exists (\textit{energy_cap_equals})) \land \textit{cap_method}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node} = \textit{energy_cap_equals}_\text{node,tech} \times \textit{energy_cap_scale}_\text{node,tech} \times \textbf{purchased}_\text{tech,node}&\quad
            if (\exists (\textit{energy_cap_equals}))
            \\
            \textbf{energy_cap}_\text{tech,node}\leq{}\textit{energy_cap_max}_\text{node,tech} \times \textit{energy_cap_scale}_\text{node,tech} \times \textbf{purchased}_\text{tech,node}&\quad
            if (\neg (\exists (\textit{energy_cap_equals})))
            \\
        \end{cases}

energy_capacity_min_purchase_milp
=================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{cost_purchase}) \land \exists (\textit{energy_cap_min}) \land \neg (\exists (\textit{energy_cap_equals})) \land \textit{cap_method}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{energy_cap}_\text{tech,node}\geq{}\textit{energy_cap_min}_\text{node,tech} \times \textit{energy_cap_scale}_\text{node,tech} \times \textbf{purchased}_\text{tech,node}&\quad
            \\
        \end{cases}

storage_capacity_max_purchase_milp
==================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{cost_purchase}) \land (\exists (\textit{storage_cap_max}) \lor \exists (\textit{storage_cap_equals})) \land \textit{cap_method}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{tech,node} = \textit{storage_cap_equals}_\text{node,tech} \times \textbf{purchased}_\text{tech,node}&\quad
            if (\exists (\textit{storage_cap_equals}))
            \\
            \textbf{storage_cap}_\text{tech,node}\leq{}\textit{storage_cap_max}_\text{node,tech} \times \textbf{purchased}_\text{tech,node}&\quad
            if (\neg (\exists (\textit{storage_cap_equals})))
            \\
        \end{cases}

storage_capacity_min_purchase_milp
==================================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{cost_purchase}) \land \exists (\textit{storage_cap_min}) \land \neg (\exists (\textit{storage_cap_equals})) \land \textit{cap_method}\mathord{=}\text{binary})
        \end{array}
        \begin{cases}
            \textbf{storage_cap}_\text{tech,node}\geq{}\textit{storage_cap_min}_\text{node,tech} \times \textbf{purchased}_\text{tech,node}&\quad
            \\
        \end{cases}

unit_capacity_systemwide_milp
=============================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            tech \in techs
            \\
            if (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer} \land (\exists (\textit{units_max_systemwide}) \lor \exists (\textit{units_equals_systemwide})) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{purchased}_\text{tech,node}) = \textit{units_equals_systemwide}_\text{tech}&\quad
            if (\exists (\textit{units_equals_systemwide}))\land{}(\textit{cap_method}\mathord{=}\text{binary})
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{units}_\text{tech,node}) = \textit{units_equals_systemwide}_\text{tech}&\quad
            if (\exists (\textit{units_equals_systemwide}))\land{}(\textit{cap_method}\mathord{=}\text{integer})
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{purchased}_\text{tech,node})\leq{}\textit{units_max_systemwide}_\text{tech}&\quad
            if (\neg (\exists (\textit{units_equals_systemwide})))\land{}(\textit{cap_method}\mathord{=}\text{binary})
            \\
            \sum\limits_{\text{node} \in \text{nodes}} (\textbf{units}_\text{tech,node})\leq{}\textit{units_max_systemwide}_\text{tech}&\quad
            if (\neg (\exists (\textit{units_equals_systemwide})))\land{}(\textit{cap_method}\mathord{=}\text{integer})
            \\
        \end{cases}

asynchronous_con_milp
=====================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{force_asynchronous_prod_con}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep})\leq{}1 - \textbf{prod_con_switch}_\text{tech,node,timestep} \times \textit{bigM}&\quad
            \\
        \end{cases}

asynchronous_prod_milp
======================

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{force_asynchronous_prod_con}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep})\leq{}\textbf{prod_con_switch}_\text{tech,node,timestep} \times \textit{bigM}&\quad
            \\
        \end{cases}

ramping_up
==========

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{energy_ramping}) \land \neg (\textit{timesteps}\mathord{=}\text{timesteps[0]}))
        \end{array}
        \begin{cases}
            \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }\leq{}\textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_prod}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_con}\mathord{=}\text{true}))
            \\
            \frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }\leq{}\textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_con}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_prod}\mathord{=}\text{true}))
            \\
            \frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} + \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep-1} + \textbf{carrier_prod}_\text{carrier,tech,node,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }\leq{}\textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{tech,node}&\quad
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_con}\mathord{=}\text{true} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
            \\
        \end{cases}

ramping_down
============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{energy_ramping}) \land \neg (\textit{timesteps}\mathord{=}\text{timesteps[0]}))
        \end{array}
        \begin{cases}
            -1 \times \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{tech,node}\leq{}\frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_prod}_\text{carrier,tech,node,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }&\quad
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_prod}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_con}\mathord{=}\text{true}))
            \\
            -1 \times \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{tech,node}\leq{}\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }&\quad
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_con}\mathord{=}\text{true} \land \neg (\textit{allowed_carrier_prod}\mathord{=}\text{true}))
            \\
            -1 \times \textit{energy_ramping}_\text{node,tech,timestep} \times \textbf{energy_cap}_\text{tech,node}\leq{}\frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep} + \textbf{carrier_prod}_\text{carrier,tech,node,timestep} }{ \textit{timestep_resolution}_\text{timestep} } - \frac{ \textbf{carrier_con}_\text{carrier,tech,node,timestep-1} + \textbf{carrier_prod}_\text{carrier,tech,node,timestep-1} }{ \textit{timestep_resolution}_\text{timestep-1} }&\quad
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_con}\mathord{=}\text{true} \land \textit{allowed_carrier_prod}\mathord{=}\text{true})
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
            node \in nodes, 
            tech \in techs, 
            cost \in costs, 
            timestep \in timesteps
            \\
            if (\exists (\textit{cost_export}) \lor \exists (\textit{cost_om_con}) \lor \exists (\textit{cost_om_prod}))
        \end{array}
        \begin{cases}
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{tech,node,timestep})&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply} \land \textit{energy_eff}\mathord{>}\text{0} \land \text{carrier_tier}\in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier}\in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_om_con})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{tech,node,timestep})&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply} \land \textit{energy_eff}\mathord{>}\text{0} \land \text{carrier_tier}\in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier}\in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_om_con})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{tech,node,timestep})&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply} \land \textit{energy_eff}\mathord{>}\text{0} \land \text{carrier_tier}\in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier}\in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_export}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carriers}} (\textbf{carrier_export}_\text{carrier,tech,node,timestep}))&\quad
            if (\exists (\textit{export_carrier}) \land \exists (\textit{cost_export}))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\neg (\exists (\textit{cost_om_con})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{tech,node,timestep})&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply} \land \textit{energy_eff}\mathord{>}\text{0} \land \text{carrier_tier}\in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\exists (\textit{cost_om_con}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier}\in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier=primary_carrier_out}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \text{tech_group=conversion_plus})\land{}(\neg (\exists (\textit{cost_om_con})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{tech,node,timestep})&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply} \land \textit{energy_eff}\mathord{>}\text{0} \land \text{carrier_tier}\in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) + \textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\exists (\textit{cost_om_con}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier}\in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_prod}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\exists (\textit{cost_om_prod}) \land \neg (\text{tech_group=conversion_plus}))\land{}(\neg (\exists (\textit{cost_om_con})))
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_con}_\text{cost,node,tech,timestep} \times \textbf{resource_con}_\text{tech,node,timestep})&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\frac{ \textit{cost_om_con}_\text{cost,node,tech,timestep} \times \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{carrier_prod}_\text{carrier,tech,node,timestep}) }{ \textit{energy_eff}_\text{node,tech,timestep} })&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=supply} \land \textit{energy_eff}\mathord{>}\text{0} \land \text{carrier_tier}\in \text{[out]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier=primary_carrier_in}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \text{tech_group=conversion_plus})
            \\
            \textit{timestep_weights}_\text{timestep} \times (\textit{cost_om_con}_\text{cost,node,tech,timestep} \times -1 \times \sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{carrier_con}_\text{carrier,tech,node,timestep}))&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_om_con}) \land \neg (\text{tech_group=conversion_plus} \lor \text{tech_group=supply_plus} \lor \text{tech_group=supply}) \land \text{carrier_tier}\in \text{[in]})
            \\
            \textit{timestep_weights}_\text{timestep} \times (0)&\quad
            if (\neg (\exists (\textit{cost_export})))\land{}(\neg (\exists (\textit{cost_om_prod})))\land{}(\neg (\exists (\textit{cost_om_con})))
            \\
        \end{cases}

cost_investment
===============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            cost \in costs
            \\
            if (\exists (\textit{cost_energy_cap}) \lor \exists (\textit{cost_om_annual}) \lor \exists (\textit{cost_om_annual_investment_fraction}) \lor \exists (\textit{cost_purchase}) \lor \exists (\textit{cost_resource_area}) \lor \exists (\textit{cost_resource_cap}) \lor \exists (\textit{cost_storage_cap}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (0) \times (1 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\neg (\text{tech_group=transmission}))\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_area}_\text{cost,node,tech} \times \textbf{resource_area}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area}))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_storage_cap}_\text{cost,node,tech} \times \textbf{storage_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage}))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{purchased}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{binary})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node} + \textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_purchase}_\text{cost,node,tech} \times \textbf{units}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\exists (\textit{cost_purchase}) \land \textit{cap_method}\mathord{=}\text{integer})\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node} + \textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_energy_cap}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\exists (\textit{cost_energy_cap}))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (\textit{cost_resource_cap}_\text{cost,node,tech} \times \textbf{resource_cap}_\text{tech,node}) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus})
            \\
            \textit{annualisation_weight} \times (\textit{cost_depreciation_rate}_\text{cost,node,tech} \times (0) \times (0.5 + \textit{cost_om_annual_investment_fraction}_\text{cost,node,tech}) + (\textit{cost_om_annual}_\text{cost,node,tech} \times \textbf{energy_cap}_\text{tech,node}))&\quad
            if (\text{tech_group=transmission})\land{}(\neg (\exists (\textit{cost_resource_area}) \land (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area})))\land{}(\neg (\exists (\textit{cost_storage_cap}) \land (\text{tech_group=supply_plus} \lor \text{tech_group=storage})))\land{}(\neg (\exists (\textit{cost_purchase}) \land (\textit{cap_method}\mathord{=}\text{binary} \lor \textit{cap_method}\mathord{=}\text{integer})))\land{}(\neg (\exists (\textit{cost_energy_cap})))\land{}(\neg (\exists (\textit{cost_resource_cap}) \land \text{tech_group=supply_plus}))
            \\
        \end{cases}

cost
====

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            cost \in costs
            \\
            if (\exists (\textit{cost_energy_cap}) \lor \exists (\textit{cost_om_annual}) \lor \exists (\textit{cost_om_annual_investment_fraction}) \lor \exists (\textit{cost_purchase}) \lor \exists (\textit{cost_resource_area}) \lor \exists (\textit{cost_resource_cap}) \lor \exists (\textit{cost_storage_cap}) \lor \exists (\textit{cost_export}) \lor \exists (\textit{cost_om_con}) \lor \exists (\textit{cost_om_prod}))
        \end{array}
        \begin{cases}
            \textbf{cost_investment}_\text{tech,node,cost} + \sum\limits_{\text{timestep} \in \text{timesteps}} (\textbf{cost_var}_\text{tech,node,timestep,cost})&\quad
            if (\exists (\textit{cost_export}) \lor \exists (\textit{cost_om_con}) \lor \exists (\textit{cost_om_prod}))\land{}(\exists (\textit{cost_energy_cap}) \lor \exists (\textit{cost_om_annual}) \lor \exists (\textit{cost_om_annual_investment_fraction}) \lor \exists (\textit{cost_purchase}) \lor \exists (\textit{cost_resource_area}) \lor \exists (\textit{cost_resource_cap}) \lor \exists (\textit{cost_storage_cap}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
            \\
            \sum\limits_{\text{timestep} \in \text{timesteps}} (\textbf{cost_var}_\text{tech,node,timestep,cost})&\quad
            if (\exists (\textit{cost_export}) \lor \exists (\textit{cost_om_con}) \lor \exists (\textit{cost_om_prod}))\land{}(\neg (\exists (\textit{cost_energy_cap}) \lor \exists (\textit{cost_om_annual}) \lor \exists (\textit{cost_om_annual_investment_fraction}) \lor \exists (\textit{cost_purchase}) \lor \exists (\textit{cost_resource_area}) \lor \exists (\textit{cost_resource_cap}) \lor \exists (\textit{cost_storage_cap}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate})))
            \\
            \textbf{cost_investment}_\text{tech,node,cost}&\quad
            if (\neg (\exists (\textit{cost_export}) \lor \exists (\textit{cost_om_con}) \lor \exists (\textit{cost_om_prod})))\land{}(\exists (\textit{cost_energy_cap}) \lor \exists (\textit{cost_om_annual}) \lor \exists (\textit{cost_om_annual_investment_fraction}) \lor \exists (\textit{cost_purchase}) \lor \exists (\textit{cost_resource_area}) \lor \exists (\textit{cost_resource_cap}) \lor \exists (\textit{cost_storage_cap}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
            \\
            0&\quad
            if (\neg (\exists (\textit{cost_export}) \lor \exists (\textit{cost_om_con}) \lor \exists (\textit{cost_om_prod})))\land{}(\neg (\exists (\textit{cost_energy_cap}) \lor \exists (\textit{cost_om_annual}) \lor \exists (\textit{cost_om_annual_investment_fraction}) \lor \exists (\textit{cost_purchase}) \lor \exists (\textit{cost_resource_area}) \lor \exists (\textit{cost_resource_cap}) \lor \exists (\textit{cost_storage_cap}) \land \neg (\text{run_config.mode}\mathord{=}\text{operate})))
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
            node \in nodes, 
            tech \in techs
            \\
            if (\neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{energy_cap_min}_\text{node,tech}\leq{}\textbf{energy_cap}_\text{tech,node}&\quad
            \\
            \textbf{energy_cap}_\text{tech,node}\leq{}\textit{energy_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

carrier_prod
============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_prod}\mathord{=}\text{true} \land \text{carrier_tier}\in \text{[out,out_2,out_3]})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{carrier_prod}_\text{carrier,tech,node,timestep}&\quad
            \\
            \textbf{carrier_prod}_\text{carrier,tech,node,timestep}\leq{}inf&\quad
            \\
        \end{cases}

carrier_con
===========

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{carrier}) \land \textit{allowed_carrier_con}\mathord{=}\text{true} \land \text{carrier_tier}\in \text{[in,in_2,in_3]})
        \end{array}
        \begin{cases}
            -inf\leq{}\textbf{carrier_con}_\text{carrier,tech,node,timestep}&\quad
            \\
            \textbf{carrier_con}_\text{carrier,tech,node,timestep}\leq{}0&\quad
            \\
        \end{cases}

carrier_export
==============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\exists (\textit{export_carrier}) \land \textit{export}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{carrier_export}_\text{carrier,tech,node,timestep}&\quad
            \\
            \textbf{carrier_export}_\text{carrier,tech,node,timestep}\leq{}inf&\quad
            \\
        \end{cases}

resource_area
=============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\exists (\textit{resource_area_min}) \lor \exists (\textit{resource_area_max}) \lor \exists (\textit{resource_area_equals}) \lor \exists (\textit{resource_area_per_energy_cap}) \lor \textit{resource_unit}\mathord{=}\text{energy_per_area} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{resource_area_min}_\text{node,tech}\leq{}\textbf{resource_area}_\text{tech,node}&\quad
            \\
            \textbf{resource_area}_\text{tech,node}\leq{}\textit{resource_area_max}_\text{node,tech}&\quad
            \\
        \end{cases}

resource_con
============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{resource_con}_\text{tech,node,timestep}&\quad
            \\
            \textbf{resource_con}_\text{tech,node,timestep}\leq{}inf&\quad
            \\
        \end{cases}

resource_cap
============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\text{tech_group=supply_plus} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{resource_cap_min}_\text{node,tech}\leq{}\textbf{resource_cap}_\text{tech,node}&\quad
            \\
            \textbf{resource_cap}_\text{tech,node}\leq{}\textit{resource_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_cap
===========

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\text{tech_group=storage} \lor \text{tech_group=supply_plus} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}) \land \textit{include_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textit{storage_cap_min}_\text{node,tech}\leq{}\textbf{storage_cap}_\text{tech,node}&\quad
            \\
            \textbf{storage_cap}_\text{tech,node}\leq{}\textit{storage_cap_max}_\text{node,tech}&\quad
            \\
        \end{cases}

storage
=======

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\text{tech_group=storage} \lor \text{tech_group=supply_plus} \land \textit{include_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{storage}_\text{tech,node,timestep}&\quad
            \\
            \textbf{storage}_\text{tech,node,timestep}\leq{}inf&\quad
            \\
        \end{cases}

purchased
=========

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\textit{cap_method}\mathord{=}\text{binary} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            0\leq{}\textbf{purchased}_\text{tech,node}&\quad
            \\
            \textbf{purchased}_\text{tech,node}\leq{}1&\quad
            \\
        \end{cases}

units
=====

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs
            \\
            if (\textit{cap_method}\mathord{=}\text{integer} \land \neg (\text{run_config.mode}\mathord{=}\text{operate}))
        \end{array}
        \begin{cases}
            \textit{units_min}_\text{node,tech}\leq{}\textbf{units}_\text{tech,node}&\quad
            \\
            \textbf{units}_\text{tech,node}\leq{}\textit{units_max}_\text{node,tech}&\quad
            \\
        \end{cases}

operating_units
===============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{cap_method}\mathord{=}\text{integer})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{operating_units}_\text{tech,node,timestep}&\quad
            \\
            \textbf{operating_units}_\text{tech,node,timestep}\leq{}inf&\quad
            \\
        \end{cases}

prod_con_switch
===============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            tech \in techs, 
            timestep \in timesteps
            \\
            if (\textit{force_asynchronous_prod_con}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{prod_con_switch}_\text{tech,node,timestep}&\quad
            \\
            \textbf{prod_con_switch}_\text{tech,node,timestep}\leq{}1&\quad
            \\
        \end{cases}

unmet_demand
============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{run_config.ensure_feasibility}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0\leq{}\textbf{unmet_demand}_\text{carrier,node,timestep}&\quad
            \\
            \textbf{unmet_demand}_\text{carrier,node,timestep}\leq{}inf&\quad
            \\
        \end{cases}

unused_supply
=============

.. container:: scrolling-wrapper

    .. math::
        
        \begin{array}{r}
            \forall{}
            node \in nodes, 
            carrier \in carriers, 
            timestep \in timesteps
            \\
            if (\text{run_config.ensure_feasibility}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -inf\leq{}\textbf{unused_supply}_\text{carrier,node,timestep}&\quad
            \\
            \textbf{unused_supply}_\text{carrier,node,timestep}\leq{}0&\quad
            \\
        \end{cases}
