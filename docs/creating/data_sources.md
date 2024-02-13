# Loading tabular data (`data_sources`)

We have chosen YAML syntax to define Calliope models as it is human-readable.
However, when you have a large dataset, the YAML files can become large and ultimately not as readable as we would like.
For instance, for parameters that vary in time we would have a list of 8760 values and timestamps to put in our YAML file!

Therefore, alongside your YAML model definition, you can load tabular data from CSV files (or from in-memory [pandas.DataFrame][] objects) under the `data_sources` top-level key.
As of Calliope v0.7.0, this tabular data can be of _any_ kind.
Prior to this, loading from file was limited to timeseries data.

The full syntax from loading tabular data can be found in the associated [schema][data-source-schema].
In brief it is:

* **source**: path to file or reference name for an in-memory object.
* **rows**: the dimension(s) in your table defined per row.
* **columns**: the dimension(s) in your table defined per column.
* **select**: values within dimensions that you want to select from your tabular data, discarding the rest.
* **drop**: dimensions to drop from your rows/columns, e.g., a "comment" row.
* **add_dimensions**: dimensions to add to the table after loading it in, with the corresponding value(s) to assign to the dimension index.

When we refer to "dimensions", we mean the sets over which data is indexed in the model: `nodes`, `techs`, `timesteps`, `carriers`, `costs`.
In addition, when loading from file, there is the _required_ dimension `parameters`.
This is a placeholder to point Calliope to the parameter name(s) that your tabular data is referring to.
The values assigned as parameters will become the array names in your loaded model dataset ([`model.inputs`][calliope.Model.inputs]).

## YAML vs tabular definitions

In this section we will show some examples of loading data and provide the equivalent YAML definition that it would replace.

### Loading timeseries data

=== "Tabular"

    Data in file:

    ```
    ,
    2005-01-01 12:00:00,15
    2005-01-01 13:00:00,5
    ...
    ```

    YAML definition to load data:

    ```yaml
    data_sources:
      pv_capacity_factor_data:
        source: data_sources/pv_resource.csv
        rows: timesteps
        add_dimensions:
          techs: pv
          parameters: source_use_equals
    ```

=== "YAML"

    ```yaml
    techs:
      pv:
        source_use_equals:
          data: [15, 5, 20, 11, ...]
          index: ["2005-01-01 12:00:00", "2005-01-01 13:00:00", "2005-01-01 12:00:00", "2005-01-01 13:00:00", ....]
          dims: "timesteps"
    ```

??? note "Getting timestamp formats right"
    By default, Calliope expects time series data in a model to be indexed by ISO 8601 compatible time stamps in the format `YYYY-MM-DD hh:mm:ss`, e.g. `2005-01-01 00:00:00`.
    This can be changed by setting `config.build.time_format` based on [`strftime` directives](https://strftime.org/), which defaults to `"ISO8601"`.

    If you work with your CSV files in Excel, keep a careful eye on the format of the timestamps.
    Excel will automatically update the format to match your operating system default, which is usually _not_ the `ISO8601` format.

### Loading technology data from file

=== "Tabular"

    Data in file:

    ```shell
    tech1,base_tech,supply  # (1)!
    tech1,flow_cap_max,100
    tech1,flow_out_eff,0.1
    tech1,area_use_max,500
    tech1,area_use_per_flow_cap,7
    tech2,base_tech,demand
    tech3,base_tech,storage
    tech3,storage_cap_max,200
    tech3,flow_cap_max,100
    ```

    1. Unlike the previous example, we do not have a "header" row with column names in this file.
    We start directly with defining data.
    Our dimensions are _only_ defined per row, not per column.

    YAML definition to load data:

    ```yaml
    data_sources:
      tech_data:
        source: data_sources/tech_data.csv
        rows: [techs, parameters]
    ```

=== "YAML"

    ```yaml
    techs:
      tech1:
        base_tech: supply
        flow_cap_max: 100
        flow_out_eff: 0.1
        area_use_max: 500
        area_use_per_flow_cap: 7
      tech2:
        base_tech: demand
      tech3:
        base_tech: storage
        storage_cap_max: 200
        flow_cap_max: 100
    ```

### Loading technology cost data from file

=== "Tabular"

    Data in file:

    ```
    tech1,cost_flow_cap,100
    tech1,cost_area_use,50
    tech1,cost_flow_out,0.2
    tech1,cost_interest_rate,0.1
    tech3,cost_flow_cap,20
    tech3,cost_storage_cap,150
    tech3,cost_interest_rate,0.1
    ```

    YAML definition to load data:

    ```yaml
    data_sources:
      tech_data:
        source: data_sources/tech_data.csv
        rows: [techs, parameters]
        add_dimensions:
          costs: monetary
    ```

=== "YAML"

    ```yaml
    tech_groups:  # (1)!
      cost_setter:
        cost_interest_rate:
          data: 0.1
          index: monetary
          dims: costs
        cost_flow_cap:
          data: null
          index: monetary
          dims: costs
        cost_area_use:
          data: null
          index: monetary
          dims: costs
        cost_flow_out:
          data: null
          index: monetary
          dims: costs
        cost_storage_cap:
          data: null
          index: monetary
          dims: costs

    techs:
      tech1:
        inherit: cost_setter
        cost_flow_cap.data: 100
        cost_area_use.data: 50
        cost_flow_out.data: 0.2
      tech3:
        cost_flow_cap.data: 20
        cost_storage_cap.data: 150
    ```

    1. To limit repetition, we have defined [technology groups](groups.md) for our costs.

!!! info "See also"
    Our [data source loading tutorial][loading-tabular-data] has more examples of loading tabular data into your model.

## Selecting dimension values and dropping dimensions

If you only want to use a subset of the tabular data you've defined, you can select it at load time.
For instance:

Data in file:

```
,,node1,node2,node3
tech1,parameter1,100,200,300
tech2,parameter1,0.1,0.3,0.5
tech3,parameter1,20,45,50
```

YAML definition to load only data from nodes 1 and 2:

```yaml
data_sources:
  tech_data:
    source: data_sources/tech_data.csv
    rows: [techs, parameters]
    columns: nodes
    select:
      nodes: [node1, node2]
```

You may also want to store scenarios in your file.
When you load in the data, you can select your scenario.
You will also need to `drop` the dimension so that it doesn't appear in the final calliope model dataset:

```
,,scenario1,scenario2
tech1,parameter1,100,200
tech2,parameter1,0.1,0.3
tech3,parameter1,20,45
```

YAML definition to load only data from scenario 1:

```yaml
data_sources:
  tech_data:
    source: data_sources/tech_data.csv
    rows: [techs, parameters]
    columns: scenarios
    select:
      scenarios: scenario1
    drop: scenarios
```

You can then also tweak just one line of your data source YAML with an [override](scenarios.md) to point to your other scenario:

```yaml
override:
  switch_to_scenario2:
    data_sources.tech_data.select.scenarios: scenario2  # (1)!
```

1. We use the dot notation as a shorthand for [abbreviate nested dictionaries](yaml.md#abbreviated-nesting).

## Adding dimensions

We used the `add_dimensions` functionality in some examples earlier in this page.
It's a useful mechanism to avoid repetition in the tabular data, and offers you the possibility to use the same data for different parts of your model definition.

For example, to define costs for the parameter `cost_flow_cap`:

=== "Without `add_dimensions`"

    ```
    ,,,node1,node2,node3
    tech1,monetary,cost_flow_cap,100,200,300
    tech2,monetary,cost_flow_cap,0.1,0.3,0.5
    tech3,monetary,cost_flow_cap,20,45,50
    ```

    ```yaml
    data_sources:
      tech_data:
        source: data_sources/tech_data.csv
        rows: [techs, costs, parameters]
        columns: nodes
    ```

=== "With `add_dimensions`"

    ```
    ,node1,node2,node3
    tech1,100,200,300
    tech2,0.1,0.3,0.5
    tech3,20,45,50
    ```

    ```yaml
    data_sources:
      tech_data:
        source: data_sources/tech_data.csv
        rows: techs
        columns: nodes
        add_dimensions:
          costs: monetary
          parameters: cost_flow_cap
    ```

Or to define the same timeseries source data for two technologies at different nodes:

=== "Without `add_dimensions`"

    ```
    ,node1,node2
    ,tech1,tech2
    ,source_use_max,source_use_max
    2005-01-01 00:00,100,100
    2005-01-01 00:00,200,200
    ...
    ```

    ```yaml
    data_sources:
      tech_data:
        source: data_sources/tech_data.csv
        rows: timesteps
        columns: [nodes, techs, parameters]
    ```



=== "With `add_dimensions`"

    ```
    2005-01-01 00:00,100
    2005-01-01 00:00,200
    ...
    ```

    ```yaml
    data_sources:
      tech_data_1:
        source: data_sources/tech_data.csv
        rows: timesteps
        add_dimensions:
          techs: tech1
          nodes: node1
          parameters: source_use_max
      tech_data_2:
        source: data_sources/tech_data.csv
        rows: timesteps
        add_dimensions:
          techs: tech2
          nodes: node2
          parameters: source_use_max
    ```

## Loading CSV files vs `pandas` dataframes

To load from CSV, set the filepath in `source` to point to your file.
This filepath can either be relative to your `model.yaml` file (as in the above examples) or an absolute path.

To load from a [pandas.DataFrame][], you can specify the `data_source_dfs` dictionary of objects when you initialise your model:

```python
import calliope
import pandas as pd
df1 = pd.DataFrame(...)
df2 = pd.DataFrame(...)

model = calliope.Model(
    "path/to/model.yaml",
    data_source_dfs={"data_source_1": df1, "data_source_2": df2}
)
```

And then you point to those dictionary keys in the `source` for your data source:

```yaml
data_sources:
  ds1:
    source: data_source_1
    ...
  ds2:
    source: data_source_2
    ...
```

!!! note
    As with loading tabular data from CSV, you will need to specify `rows`, `columns`, etc. based on the shape of your dataframe.
    Rows correspond to your dataframe index levels and columns to your dataframe column levels.

    You _cannot_ specify [pandas.Series][] objects.
    Ensure you convert them to dataframes (`to_frame()`) before adding them to your data source dictionary.

## Important considerations

To get the expected results when loading tabular data, here are some things to note:

1. You must always have a `parameters` dimension.
This could be defined in `rows`, `columns`, or `add_dimensions`.
2. The order of events for `select`, `drop`, `add_dimensions` is:
    1. `select` from dimensions;
    2. `drop` unwanted dimensions;
    3. `add_dimensions` to add dimensions.
This means you can technically select value "A" from dimensions `nodes`, then drop `nodes`, then add `nodes` back in with the value "B".
This effectively replaces "A" with "B" on that dimension.
3. The order of tabular data loading is in the order you list the sources.
If a new table has data which clashes with preceding data sources, it will override that data.
This may have unexpected results if the files have different dimensions as the dimensions will be broadcast to match each other.
4. CSV files must have `.csv` in their filename (even if compressed, e.g., `.csv.zip`).
If they don't, they won't be picked up by Calliope.
5. We automatically infer which technologies are available at which nodes according to any tabular data containing _both_ the `nodes` and `techs` dimensions.
However, we do not recommend you rely on tabular data entirely to define your model.
Instead, at least list the techs available at each node in YAML.
E.g.,
    ```yaml
    nodes:
      node1.techs: {tech1, tech2, tech3}
      node2.techs: {tech1, tech2}
    data_sources:
      ...
    ```

### Data you _cannot_ load in tabular format

Some data is specific to the YAML definition or is computed by Calliope internally and therefore cannot be loaded by the user from tabular data.
These are:

```yaml
--8<-- "src/calliope/config/protected_parameters.yaml"
```
