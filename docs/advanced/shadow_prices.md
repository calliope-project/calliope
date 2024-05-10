# Shadow prices

In a linear problem, you can obtain the [shadow prices](https://en.wikipedia.org/wiki/Shadow_price) (dual variables) for each constraint from the Pyomo backend.
This can prove particularly useful if you are linking Calliope with other models or want to dig deeper into the economic impacts of your designed energy system.

You can access shadow prices by specifying a list of constraints for which shadow prices should be returned in the results, in your `solve` configuration:

```yaml
config:
    solve:
        shadow_prices: ["system_balance", ...]
```

A list of the available constraint names can be found under "Subject to" in [our base math documentation page][base-math].
If you [define any of your own math constraints](../user_defined_math/components.md#constraints), you can also reference those by name in the list.

!!! note

    * Not all solvers provide access to shadow prices.
    For instance, we know that it is possible with Gurobi and GLPK, but not with CBC.
    Since we cannot test all Pyomo-compatible solvers, you may run into issues depending on the solver you use.
    * You cannot access shadow prices if you have integer/binary variables in your model.
    If you try to do so, you will receive a warning, and shadow price tracking will remain disabled.
    * You can check the status of shadow price tracking with `model.backend.shadow_prices.is_active`.

## Shadow prices when using the command-line tool

By specifying constraints in the YAML configuration (see above), shadow price tracking will be activated and the shadow prices of those constraints you have listed will be available in the results dataset, prefixed with `shadow_price_`.
For instance, listing `system_balance` in the configuration will lead to `shadow_price_system_balance` being available in the optimisation results that are saved to file on calling [`calliope run ...`](../running.md#running-with-the-command-line-tool).

## Shadow prices when running in Python

When running in Python, you can additionally turn on shadow price tracking by running `model.backend.shadow_prices.activate()` after `model.build()`.
By doing that, or by having added at least one valid constraint in `config.solve.shadow_prices` (see above), shadow price tracking will be enabled.

Then, you can access shadow prices for any constraint once you have an optimal solution by running `model.backend.shadow_prices.get("constraint_name")`, which returns an [xarray.DataArray][].
As with [running in the command-line tool](#shadow-prices-when-using-the-command-line-tool), having a list of shadow prices to track in the solve configuration will lead to shadow prices being available automatically in the model [results dataset][calliope.Model.results].

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

    Or:

    ```python
    model = calliope.examples.national_scale()
    model.build()
    model.solve(shadow_prices=["system_balance"])

    balance_price = model.results.shadow_price_system_balance.to_series()
    ```
