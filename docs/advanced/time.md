
# Time adjustment

## Time resolution adjustment

Models have a default timestep length (defined implicitly by the timesteps of the model's time series data).
This default resolution can be adjusted by specifying time resolution adjustment in the model configuration, for example:

```yaml
config:
    init:
        time_resample: 6H
```

In the above example, this would resample all time series data to 6-hourly timesteps.
Any [pandas-compatible rule describing the target resolution][pandas.DataFrame.resample] can be used.

## Time clustering
<!-- TODO -->