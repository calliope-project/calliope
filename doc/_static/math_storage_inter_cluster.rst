

Subject to
----------

storage_max
^^^^^^^^^^^

:red:`REMOVED`


balance_supply_plus_with_storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:yellow:`UPDATED`
Set the upper bound on, or a fixed total of, a `supply_plus` (with storage) technology's ability to produce flow based on the quantity of consumed resource and available stored carrier.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ carrier }\negthickspace \in \negthickspace\text{ carriers, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\exists (\textbf{storage}_\text{node,tech,timestep}) \land \text{tech_group=supply_plus})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech} + (\textbf{source_use}_\text{node,tech,timestep} \times \textit{source_eff}_\text{node,tech}) - (\frac{ \textbf{flow_out}_\text{node,tech,carrier,timestep} }{ (\textit{flow_out_eff}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech}) })&\quad
            \text{if } (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} + (\textbf{source_use}_\text{node,tech,timestep} \times \textit{source_eff}_\text{node,tech}) - (\frac{ \textbf{flow_out}_\text{node,tech,carrier,timestep} }{ (\textit{flow_out_eff}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech}) })&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep})))
            \\
            \textbf{storage}_\text{node,tech,timestep} = (\textbf{source_use}_\text{node,tech,timestep} \times \textit{source_eff}_\text{node,tech}) - (\frac{ \textbf{flow_out}_\text{node,tech,carrier,timestep} }{ (\textit{flow_out_eff}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech}) })&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep}) \land \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true})))
            \\
        \end{cases}

balance_storage
^^^^^^^^^^^^^^^

:yellow:`UPDATED`
Fix the quantity of carrier stored in a `storage` technology at the end of each timestep based on the net flow of carrier charged and discharged and the quantity of carrier stored at the start of the timestep.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \text{if } (\text{tech_group=storage})
        \end{array}
        \begin{cases}
            \textbf{storage}_\text{node,tech,timestep} = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech} - (\frac{ \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{flow_out}_\text{node,tech,carrier,timestep}) }{ (\textit{flow_out_eff}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech}) }) + (\sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{flow_in}_\text{node,tech,carrier,timestep}) \times \textit{flow_in_eff}_\text{node,tech})&\quad
            \text{if } (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage}_\text{node,tech,timestep} = ((1 - \textit{storage_loss}_\text{node,tech})^{\textit{timestep_resolution}_\text{timestep-1}}) \times \textbf{storage}_\text{node,tech,timestep-1} - (\frac{ \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{flow_out}_\text{node,tech,carrier,timestep}) }{ (\textit{flow_out_eff}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech}) }) + (\sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{flow_in}_\text{node,tech,carrier,timestep}) \times \textit{flow_in_eff}_\text{node,tech})&\quad
            \text{if } (((\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \text{config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]})) \land \neg (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep})))
            \\
            \textbf{storage}_\text{node,tech,timestep} = (\frac{ \sum\limits_{\text{carrier} \in \text{carrier_tier(out)}} (\textbf{flow_out}_\text{node,tech,carrier,timestep}) }{ (\textit{flow_out_eff}_\text{node,tech} \times \textit{parasitic_eff}_\text{node,tech}) }) + (\sum\limits_{\text{carrier} \in \text{carrier_tier(in)}} (\textbf{flow_in}_\text{node,tech,carrier,timestep}) \times \textit{flow_in_eff}_\text{node,tech})&\quad
            \text{if } (\exists (\textit{lookup_cluster_last_timestep}_\text{timestep}) \land \neg (\textit{timesteps}_\text{timestep}\mathord{=}\text{timesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true})))
            \\
        \end{cases}

set_storage_initial
^^^^^^^^^^^^^^^^^^^

:yellow:`UPDATED`
Fix the relationship between carrier stored in a `storage` technology at the start and end of the whole model period.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs }
            \\
            \text{if } (\exists (\textbf{storage}_\text{node,tech,timestep}) \land \exists (\textit{storage_initial}_\text{node,tech}) \land \text{config.cyclic_storage}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            \textbf{storage_inter_cluster}_\text{node,tech,datestep=datesteps[-1]} \times ((1 - \textit{storage_loss}_\text{node,tech})^{24}) = \textit{storage_initial}_\text{node,tech} \times \textbf{storage_cap}_\text{node,tech}&\quad
            \\
        \end{cases}

balance_storage_inter
^^^^^^^^^^^^^^^^^^^^^

:green:`NEW`
Fix the relationship between one day and the next of a `storage` technology's available stored carrier, according to the previous day's representative storage fluctuations and the excess stored carrier available from all days up to this day.

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
            \text{if } (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true}))\land{}(\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = \textit{storage_initial}_\text{node,tech} + \textbf{storage}_\text{node,tech,timestep=\textit{lookup_datestep_last_cluster_timestep}_\text{datestep-1}}&\quad
            \text{if } (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true}))\land{}(\neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true})))
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = ((1 - \textit{storage_loss}_\text{node,tech})^{24}) \times \textbf{storage_inter_cluster}_\text{node,tech,datestep-1}&\quad
            \text{if } ((\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \text{config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]}))\land{}(\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true}))
            \\
            \textbf{storage_inter_cluster}_\text{node,tech,datestep} = ((1 - \textit{storage_loss}_\text{node,tech})^{24}) \times \textbf{storage_inter_cluster}_\text{node,tech,datestep-1} + \textbf{storage}_\text{node,tech,timestep=\textit{lookup_datestep_last_cluster_timestep}_\text{datestep-1}}&\quad
            \text{if } ((\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \text{config.cyclic_storage}\mathord{=}\text{true}) \lor \neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]}))\land{}(\neg (\textit{datesteps}_\text{datestep}\mathord{=}\text{datesteps[0]} \land \neg (\text{config.cyclic_storage}\mathord{=}\text{true})))
            \\
        \end{cases}

storage_inter_max
^^^^^^^^^^^^^^^^^

:green:`NEW`
Set the upper bound of a `storage` technology's stored carrier across all days in the timeseries

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
Set the lower bound of a `storage` technology's stored carrier across all days in the timeseries

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
            (\textbf{storage_inter_cluster}_\text{node,tech,datestep} \times ((1 - \textit{storage_loss}_\text{node,tech})^{24})) + \textbf{storage_intra_cluster_min}_\text{node,tech,cluster=\textit{lookup_datestep_cluster}_\text{datestep}} \geq 0&\quad
            \\
        \end{cases}

storage_intra_max
^^^^^^^^^^^^^^^^^

:green:`NEW`
Set the upper bound of a `storage` technology's stored carrier within a clustered day

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
Set the lower bound of a `storage` technology's stored carrier within a clustered day

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
The virtual carrier stored by a `supply_plus` or `storage` technology in each timestep of a clustered day. Stored carrier can be negative so long as it does not go below the carrier stored in `storage_inter_cluster`. Only together with `storage_inter_cluster` does this variable's values gain physical significance.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ timestep }\negthickspace \in \negthickspace\text{ timesteps }
            \\
            \forall\mathbb{R}\;
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
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
The virtual carrier stored by a `supply_plus` or `storage` technology between days of the entire timeseries. Only together with `storage` does this variable's values gain physical significance.

.. container:: scrolling-wrapper

    .. math::
        \begin{array}{r}
            \forall{}
            \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
            \text{ tech }\negthickspace \in \negthickspace\text{ techs, }
            \text{ datestep }\negthickspace \in \negthickspace\text{ datesteps }
            \\
            \forall\mathbb{R}\;
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
            \forall\mathbb{R}\;
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
            \forall\mathbb{R}\;
            \text{if } (\textit{include_storage}_\text{node,tech}\mathord{=}\text{true})
        \end{array}
        \begin{cases}
            -inf \leq \textbf{storage_intra_cluster_min}_\text{node,tech,cluster}&\quad
            \\
            \textbf{storage_intra_cluster_min}_\text{node,tech,cluster} \leq inf&\quad
            \\
        \end{cases}
