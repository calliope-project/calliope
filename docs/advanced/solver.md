# Specifying custom solver options

## Gurobi

Refer to the [Gurobi manual](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html), which contains a list of parameters.
Simply use the parameter names given in the documentation (e.g. "NumericFocus" to set the numerical focus value). Below we list our recommended parameters to speed-up solver performance in large models, based on insights from [empirical tests we carried out](https://doi.org/10.1186/s13705-024-00458-z).

```yaml
config.solve:
  solver: gurobi
  solver_options:
    Threads: 6            # Number of threads
    Method: 2             # Use Barrier algorithm, do not run other algorithms in parallel
    Crossover: 0          # Stop after barrier, do not perform crossover
    BarConvTol: 1e-4      # Tolerance for convergence

```

## HiGHS

When using the [HiGHS backend](backend_choice.md), refer to the [HiGHS options list](https://ergo-code.github.io/HiGHS/stable/options/definitions/) for available parameters.
For example:

```yaml
config:
  build.backend: highs
  solve.solver_options:
    threads: 6            # Number of threads
    solver: ipm           # Use interior point method rather than simplex
    run_crossover: "off"  # Stop after barrier, do not perform crossover
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
