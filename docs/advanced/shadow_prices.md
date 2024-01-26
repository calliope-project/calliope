# Shadow prices

In a linear problem, you can obtain the shadow prices (dual variables) for each constraint from the Pyomo backend.

Currently, the only way to do so is by running Calliope in Python, and not through the command-line interface.

You need to enable the tracking of shadow prices after the model is built and before it is solved by running `model.backend.shadow_prices.activate()`:

```python

model = calliope.examples.national_scale()
model.build()
model.backend.shadow_prices.activate()
model.solve()
```

Then, after solving the model successfully, you can get the shadow price(s) for any constraint by running `model.backend.shadow_prices.get("constraint_name")`, which returns a `pandas.DataFrame`. For example, to get the shadow prices of the system_balance constraint indexed over nodes, carriers and timesteps:

```python
balance_price = model.backend.shadow_prices.get("system_balance").to_series()
```
