# Specifying custom solver options

## Gurobi

Refer to the [Gurobi manual](https://www.gurobi.com/documentation/), which contains a list of parameters.
Simply use the names given in the documentation (e.g. "NumericFocus" to set the numerical focus value).
For example:

```yaml
config.solve:
    solver: gurobi
    solver_options:
        Threads: 3
        NumericFocus: 2
```

## CPLEX

Refer to the [CPLEX parameter list](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-list-parameters).
Use the "Interactive" parameter names, replacing any spaces with underscores (e.g., the memory reduction switch is called "emphasis memory", and thus becomes "emphasis_memory").
For example:

```yaml
config.solve:
    solver: cplex
    solver_options:
        mipgap: 0.01
        mip_polishafter_absmipgap: 0.1
        emphasis_mip: 1
        mip_cuts: 2
        mip_cuts_cliques: 3
```
