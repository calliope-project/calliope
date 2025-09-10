# Specifying custom solver options

## Gurobi

Refer to the [Gurobi manual](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html), which contains a list of parameters.
Simply use the names given in the documentation (e.g. "NumericFocus" to set the numerical focus value). We report below, as an illustrative example, the typical parameters that we used to speed-up the performance of the solver, especially for large models, based on the insights from [empirical tests we carried out](https://doi.org/10.1186/s13705-024-00458-z).

```yaml
config.solve:
  solver: gurobi
  solver_options:
    Threads: 6            # Number of threads
    Method: 2             # Use Barrier algorithm, do not run other algorithms in parallel
    Crossover: 0          # Stop after barrier, do not perform crossover
    BarConvTol: 1e-4      # Tolerance for convergence

```

## CPLEX

Refer to the [CPLEX parameter list](https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-list-parameters).
Use the "Interactive" parameter names, replacing any spaces with underscores (e.g., the memory reduction switch is called "emphasis memory", and thus becomes "emphasis_memory").
For example, a similar configuration to the one illustrated above for Gurobi, would look as follows:

```yaml
config.solve:
  solver: cplex
  solver_options:
    threads: 6                    # Number of threads
    lpmethod: 4                   # Use Barrier algorithm, do not run other algorithms in parallel
    solutiontype: 2               # Stop after barrier, do not perform crossover
    barrier_convergetol: 1e-4     # Tolerance for convergence
```
