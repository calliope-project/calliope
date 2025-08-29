
# Scenarios and overrides

You may want to define slightly different data for sensitivity analyses, or to compare the effect of resampling your time dimension to varying degrees.
There's no need to create different model files for each of these. Instead, you can define overrides and scenarios in your main model definition:

- `overrides` are blocks of YAML that specify configurations that expand or override parts of the base model.
- `scenarios` are combinations of any number of such overrides.

Both are specified at the top level of the model configuration, as in this example `model.yaml` file:

```yaml
scenarios:
  high_cost_2005: ["high_cost", "year2005"]
  high_cost_2006: ["high_cost", "year2006"]

overrides:
  high_cost:
    techs.onshore_wind.cost_flow_cap.data: 2000
  year2005:
    init.subset.timesteps: ['2005-01-01', '2005-12-31']
  year2006:
    init.subset.timesteps: ['2006-01-01', '2006-12-31']

config:
    ...
```

Each override is given by a name (e.g. `high_cost`) and any number of model settings - anything in the model configuration can be overridden by an override.
In the above example, one override defines higher costs for an `onshore_wind` technology.
The other two other overrides specify different time subsets, so would run an otherwise identical model over two different periods of timeseries data.

One or several overrides can be applied when [running a model](../running.md).
Overrides can also be combined into scenarios to make applying them at run-time easier.
Scenarios consist of a name and a list of override names which together form that scenario.

Scenarios and overrides can be used to generate scripts that run a single Calliope model many times, either sequentially, or in parallel on a high-performance cluster
(see the section on [generating scripts to repeatedly run variations of a model](../advanced/scripts.md)).

???+ warning
    Overrides are executed _after_ `imports:` but _before_ `templates:`.
    This means it is possible to override template values, but not the files imported in your model definition.
