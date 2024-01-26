# Shadow prices

In a linear problem, you can obtain the [shadow prices](https://en.wikipedia.org/wiki/Shadow_price) (dual variables) for each constraint from the Pyomo backend.
This can prove particularly useful if you are linking Calliope with other models or want to dig deeper into the economic impacts of your designed energy system.

You can access shadow prices for any constraint by running once you have an optimal solution by running `model.backend.shadow_prices.get("constraint_name")`, which returns an [xarray.DataArray][].

!!! example

    ```python
    model = calliope.examples.national_scale()
    model.build()
    model.backend.shadow_prices.activate()  # (1)!
    model.solve()

    balance_price = model.backend.shadow_prices.get("system_balance").to_series()
    ```

    1. With the Pyomo backend interface, tracking shadow prices can be memory-intensive.
    Therefore, you must actively activate tracking before solving your model.

!!! note

    * Not all solvers provide an API to access duals.
    For instance, we know it is not possible with the CBC solver.
    * You cannot access shadow prices if you have integer/binary variables in your model.
    If you try to do so, you will receive a None/NaN-filled array.
    * You can check the status of shadow price tracking with `model.backend.shadow_prices.is_active`.
    * Currently, the only way to access shadow prices is by running Calliope in Python, and not through the command-line interface.