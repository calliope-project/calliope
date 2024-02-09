
# Inheriting from technology node groups: `tech_groups`, `node_groups`

For larger models, duplicate entries can start to crop up and become cumbersome.
To streamline data entry, technologies and nodes can inherit common data from a `tech_group` or `node_group`, respectively.

For example, if we want to set interest rate to `0.1` across all our technologies, we could define:

```yaml
tech_groups:
  interest_rate_setter:
    cost_interest_rate:
      data: 0.1
      index: monetary
      dims: costs
techs:
  ccgt:
    inherit: interest_rate_setter
    ...
  ac_transmission:
    inherit: interest_rate_setter
    ...
```

Similarly, if we want to allow the same technologies at all our nodes:

```yaml
node_groups:
  standard_tech_list:
    techs: {ccgt, battery, demand_power}  # (1)!
nodes:
  region1:
    inherit: standard_tech_list
    ...
  region2:
    inherit: standard_tech_list
    ...
  ...
  region100:
    inherit: standard_tech_list
```

1. this YAML syntax is shortform for:
    ```yaml
    techs:
      ccgt:
      battery:
      demand_power:
    ```

Inheritance chains can also be set up.
That is, groups can inherit from groups.
E.g.:

```yaml
tech_groups:
  interest_rate_setter:
    cost_interest_rate:
      data: 0.1
      index: monetary
      dims: costs
  investment_cost_setter:
    inherit: interest_rate_setter
    cost_flow_cap:
      data: 100
      index: monetary
      dims: costs
    cost_area_use:
       data: 1
       index: monetary
       dims: costs
techs:
  ccgt:
    inherit: investment_cost_setter
    ...
  ac_transmission:
    inherit: interest_rate_setter
    ...
```

Finally, inherited properties can always be overridden by the inheriting component.
This can be useful to streamline setting costs, e.g.:

```yaml
tech_groups:
  interest_rate_setter:
    cost_interest_rate:
      data: 0.1
      index: monetary
      dims: costs
  investment_cost_setter:
    inherit: interest_rate_setter
    cost_interest_rate.data: 0.2  # this will replace `0.1` in the `interest_rate_setter`.
    cost_flow_cap:
      data: null
      index: monetary
      dims: costs
    cost_area_use:
      data: null
      index: monetary
      dims: costs
techs:
  ccgt:
    inherit: investment_cost_setter
    cost_flow_cap.data: 100  # this will replace `null` in the `investment_cost_setter`.
    ...
```
