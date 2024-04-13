# Shadow prices

In a linear problem, you can obtain the [shadow prices](https://en.wikipedia.org/wiki/Shadow_price) (dual variables) for each constraint from the Pyomo backend.
This can prove particularly useful if you are linking Calliope with other models or want to dig deeper into the economic impacts of your designed energy system.

You can access shadow prices by specifying a list of constraints for which shadow prices should be returned in the results, in your `solve` configuration:

```yaml
config:
    solve:
        shadow_prices: ["system_balance", ...]
```

!!! note

    * Not all solvers provide access to shadow prices.
    For instance, we know that it is possible with Gurobi and GLPK, but not with CBC.
    Since we cannot test all Pyomo-compatible solvers, you may run into issues depending on the solver you use.
    * You cannot access shadow prices if you have integer/binary variables in your model.
    If you try to do so, you will receive a warning, and shadow price tracking will remain disabled.
    * You can check the status of shadow price tracking with `model.backend.shadow_prices.is_active`.

## Shadow prices when running in Python

When running in Python, you can turn on shadow price tracking by running `model.backend.shadow_prices.activate()` after `model.build()`. Then, you can access shadow prices for any constraint once you have an optimal solution by running `model.backend.shadow_prices.get("constraint_name")`, which returns an [xarray.DataArray][].

!!! example

    ```python
    model = calliope.examples.national_scale()
    model.build()
    model.backend.shadow_prices.activate()  # (1)!
    model.solve()

    balance_price = model.backend.shadow_prices.get("system_balance").to_series()
    ```

    1. With the Pyomo backend interface, tracking shadow prices can be memory-intensive.
    Therefore, you must manually activate tracking before solving your model.
