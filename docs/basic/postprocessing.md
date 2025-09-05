# Postprocessing

Calliope applies two postprocessing steps, which are described in turn below.

## Additional result variables

First, the following additional result variables are generated:

* `capacity_factor`: Capacity factors indexed by techs, nodes, and timesteps.
* `systemwide_capacity_factor`: Per-tech capacity factor averaged across nodes and time, indexed by techs.
  The weight of timesteps is considered when computing systemwide capacity factors, such that higher-weighted timesteps have a stronger influence on the resulting system-wide time-averaged capacity factor.
* `systemwide_levelised_cost`: The per-tech levelised cost of carrier production indexed by techs, carriers and costs (cost classes)
* `total_levelised_cost`: The carrier-total levelised cost of carrier production indexed by carriers and costs (cost classes).
* `unmet_sum`: sums up `unmet_demand` and `unmet_supply`.

Levelised costs are calculated by dividing cost by production: `cost / production`.
The production is based on `flow_out` + `flow_export` and is (temporarily, for calculation purposes only) scaled by weights to be consistent with the model.
The costs are the `cost` expression from the model results.
Costs are multiplied by weight in the constraints, so not further adjusted here.
For the exact implementation, refer to the [systemwide_levelised_cost][calliope.postprocess.systemwide_levelised_cost] function.

!!! tip
    To disable the first part of postprocessing, set `config.solve.postprocessing_active` to `false`.
    To disable the second (`zero_threshold`) of postprocessing, set `zero_threshold` to `0`.

## Zero threshold

Second, the configured `zero_threshold`  is applied: any value coming out of the backend that is smaller than this (due to floating point errors, probably) will be set to zero.
The default for `zero_threshold` is `1e-10` but by setting it to `0`, it can be disabled entirely.
