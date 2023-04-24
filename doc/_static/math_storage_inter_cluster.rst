

Subject to
----------

storage_max
^^^^^^^^^^^

:red:`REMOVED`


balance_supply_plus_with_storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:yellow:`UPDATED`
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
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep})))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} + (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep})))\land{}(\neg (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0}))
            \\
            \textbf{storage}_\text{node,tech,timestep} = (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep})&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep}))\land{}(\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0})
            \\
            \textbf{storage}_\text{node,tech,timestep} = (\textbf{resource_con}_\text{node,tech,timestep} \times \textit{resource_eff}_\text{node,tech,timestep}) - \frac{ \textbf{carrier_prod}_\text{node,tech,carrier,timestep} }{ (\textit{energy_eff}_\text{node,tech,timestep} \times \textit{parasitic_eff}_\text{node,tech,timestep}) }&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep}))\land{}(\neg (\textit{energy_eff}_\text{node,tech,timestep}\mathord{=}\text{0} \lor \textit{parasitic_eff}_\text{node,tech,timestep}\mathord{=}\text{0}))
            \\
        \end{cases}

set_storage_initial
^^^^^^^^^^^^^^^^^^^

:yellow:`UPDATED`
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
            \textbf{storage_inter_cluster}_\text{node,tech,datestep=datesteps[-1]} \times ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{24}) = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

balance_storage_inter
^^^^^^^^^^^^^^^^^^^^^

:green:`NEW`
Fix the relationship between one day and the next of a `storage` technology's available stored energy, according to the previous day's representative storage fluctuations and the excess stored energy available from all days up to this day.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ datestep }\negthickspace \in \negthickspace\text{ datesteps }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = \textit{storage_initial}_\text{node,tech}&\quad
            \text{if } (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))\land{}(\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{24}) \times \textbf{storage_inter_cluster}_\text{node,tech,datestep-1}&\quad
            \text{if } (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))\land{}((\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]}))
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = \textit{storage_initial}_\text{node,tech} + \textbf{storage}_\text{node,tech,timestep=\textit{lookup_datestep_last_cluster_timestep}_\text{datestep-1}}&\quad
            \text{if } (\neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true})))\land{}(\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{24}) \times \textbf{storage_inter_cluster}_\text{node,tech,datestep-1} + \textbf{storage}_\text{node,tech,timestep=\textit{lookup_datestep_last_cluster_timestep}_\text{datestep-1}}&\quad
            \text{if } (\neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{run_config.cyclic_storage}\mathord{=}\text{true})))\land{}((\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \text{run_config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]}))
            \\
        \end{cases}

storage_inter_max
^^^^^^^^^^^^^^^^^

:green:`NEW`
Set the upper bound of a `storage` technology's stored energy across all days in the timeseries

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ datestep }\negthickspace \in \negthickspace\text{ datesteps }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} + \textbf{storage_intra_cluster_max}_\text{node,tech,cluster=\textit{lookup_datestep_cluster}_\text{datestep}} \leq \textbf{storage_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

storage_inter_min
^^^^^^^^^^^^^^^^^

:green:`NEW`
Set the lower bound of a `storage` technology's stored energy across all days in the timeseries

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ datestep }\negthickspace \in \negthickspace\text{ datesteps }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            (\textbf{storage_inter_cluster}_\text{node,tech,datestep} \times ((1 - \textit{storage_loss}_\text{node,tech,timestep})^{24})) + \textbf{storage_intra_cluster_min}_\text{node,tech,cluster=\textit{lookup_datestep_cluster}_\text{datestep}} \geq 0&\quad
            \\
        \end{cases}

storage_intra_max
^^^^^^^^^^^^^^^^^

:green:`NEW`
Set the upper bound of a `storage` technology's stored energy within a clustered day

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
            \textbf{storage}_\text{node,tech,timestep} \leq \textbf{storage_intra_cluster_max}_\text{node,tech,cluster=\textit{timestep_cluster}_\text{timestep}}&\quad
            \\
        \end{cases}

storage_intra_min
^^^^^^^^^^^^^^^^^

:green:`NEW`
Set the lower bound of a `storage` technology's stored energy within a clustered day

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
            \textbf{storage}_\text{node,tech,timestep} \geq \textbf{storage_intra_cluster_min}_\text{node,tech,cluster=\textit{timestep_cluster}_\text{timestep}}&\quad
            \\
        \end{cases}

Decision Variables
------------------

storage
^^^^^^^

:yellow:`UPDATED`
The virtual energy stored by a `supply_plus` or `storage` technology in each timestep of a clustered day. Stored energy can be negative so long as it does not go below the energy stored in `storage_inter_cluster`. Only together with `storage_inter_cluster` does this variable's values gain physical significance.

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
            -inf \leq \textbf{storage}_\text{node,tech,timestep}&\quad
            \\
            \textbf{storage}_\text{node,tech,timestep} \leq inf&\quad
            \\
        \end{cases}

storage_inter_cluster
^^^^^^^^^^^^^^^^^^^^^

:green:`NEW`
The virtual energy stored by a `supply_plus` or `storage` technology between days of the entire timeseries. Only together with `storage` does this variable's values gain physical significance.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ datestep }\negthickspace \in \negthickspace\text{ datesteps }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            0 \leq \textbf{storage_inter_cluster}_\text{node,tech,datestep}&\quad
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} \leq inf&\quad
            \\
        \end{cases}

storage_intra_cluster_max
^^^^^^^^^^^^^^^^^^^^^^^^^

:green:`NEW`
Virtual variable to limit the maximum value of `storage` in a given representative day.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ cluster }\negthickspace \in \negthickspace\text{ clusters }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -inf \leq \textbf{storage_intra_cluster_max}_\text{node,tech,cluster}&\quad
            \\
            \textbf{storage_intra_cluster_max}_\text{node,tech,cluster} \leq inf&\quad
            \\
        \end{cases}

storage_intra_cluster_min
^^^^^^^^^^^^^^^^^^^^^^^^^

:green:`NEW`
Virtual variable to limit the minimum value of `storage` in a given representative day.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ cluster }\negthickspace \in \negthickspace\text{ clusters }
            \\
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -inf \leq \textbf{storage_intra_cluster_min}_\text{node,tech,cluster}&\quad
            \\
            \textbf{storage_intra_cluster_min}_\text{node,tech,cluster} \leq inf&\quad
            \\
        \end{cases}
