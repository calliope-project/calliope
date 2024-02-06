
# Time adjustment

## Time resolution adjustment (resampling)

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

By supplying a file linking dates in your model timeseries with representative days, it is possible to cluster your timeseries:

```yaml
config:
  init:
    time_cluster: cluster_days.csv
```

When using representative days, you may want to enable a number of additional constraints to improve how carriers are stored between representative days, based on the study undertaken by [Kotzur et al.](https://doi.org/10.1016/j.apenergy.2018.01.023).
These constraints require a new decision variable `storage_inter_cluster`, which tracks storage between all the dates of the original timeseries.
This particular math - detailed [here][inter-cluster-storage-math] - can be enabled by including `storage_inter_cluster` in your list of custom math.

We no longer provide the functionality to infer representative days from your timeseries.
Instead, we recommend you use other timeseries processing tools applied to your input CSV data or your built model dataset (`model.inputs`).

!!! note

    Resampling and clustering can be applied together.
    Resampling of your timeseries will take place _before_ clustering.

!!! warning

    When using time clustering, the resulting timesteps will be assigned different weights depending on how long a period of time they represent.
    Weights are used for example to give appropriate weight to the operational costs of aggregated typical days in comparison to individual extreme days, if both exist in the same processed time series.
    The weighting is accessible in the model data, e.g. through `#!python model.inputs.timestep_weights`.

    The interpretation of results when weights are not 1 for all timesteps requires caution.
    Production values are not scaled according to weights, but costs are multiplied by weight, in order to weight different timesteps appropriately in the objective function.
    This means that costs and outflow values are not consistent without manually post-processing them by either multiplying outflows by weight (outflows would then be inconsistent with capacities) or dividing costs by weight.
    The computation of levelised costs and of capacity factors takes weighting into account, so these values are consistent and can be used as usual.
