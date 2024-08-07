
# Inheriting from templates: `templates`

For larger models, duplicate entries can start to crop up and become cumbersome.
To streamline data entry, technologies and nodes can inherit common data from a `template`.

For example, if we want to set interest rate to `0.1` across all our technologies, we could define:

```yaml
templates:
  interest_rate_setter:
    cost_interest_rate:
      data: 0.1
      index: monetary
      dims: costs
techs:
  ccgt:
    template: interest_rate_setter
    ...
  ac_transmission:
    template: interest_rate_setter
    ...
```

Similarly, if we want to allow the same technologies at all our nodes:

```yaml
templates:
  standard_tech_list:
    techs: {ccgt, battery, demand_power}  # (1)!
nodes:
  region1:
    template: standard_tech_list
    ...
  region2:
    template: standard_tech_list
    ...
  ...
  region100:
    template: standard_tech_list
```

1. this YAML syntax is shortform for:
    ```yaml
    techs:
      ccgt:
      battery:
      demand_power:
    ```

Inheritance chains can also be set up.
That is, templates can inherit from other templates.
E.g.:

```yaml
templates:
  interest_rate_setter:
    cost_interest_rate:
      data: 0.1
      index: monetary
      dims: costs
  investment_cost_setter:
    template: interest_rate_setter
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
    template: investment_cost_setter
    ...
  ac_transmission:
    template: interest_rate_setter
    ...
```

Finally, inherited properties can always be overridden by the inheriting component.
This can be useful to streamline setting costs, e.g.:

```yaml
templates:
  interest_rate_setter:
    cost_interest_rate:
      data: 0.1
      index: monetary
      dims: costs
  investment_cost_setter:
    template: interest_rate_setter
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
    template: investment_cost_setter
    cost_flow_cap.data: 100  # this will replace `null` in the `investment_cost_setter`.
    ...
```
