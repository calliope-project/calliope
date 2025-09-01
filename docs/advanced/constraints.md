# Advanced constraints

On this page, we look at some of the more advanced features of Calliope's math and configuration that can help you understand the breadth of what we offer.

!!! info "See also"
    [Pre-defined math formulation][base-math] (which includes a description of our pre-defined parameters),
    [Model definition schema][model-definition-schema],
    [Introducing your own math to your model](../user_defined_math/customise.md),
    ["MILP" example model](../examples/milp/index.md).

## Multiple input/output carriers

Most technologies will define one carrier in and/or out.
However, there exist technologies that co-produce different carriers or have the option to switch between consuming/producing different carriers.
For example:

* Combined heat and power (CHP) plants - one carrier in, two co-produced carriers out.
* Heat pumps - one carrier in, the option to output cooling or heating.
* Partially retrofitted coal power plants - possibility to consume coal or biofuel, electricity as the only carrier out.
* Tracking auxiliary flows - nuclear power plants produce electricity and nuclear waste.

As of Calliope v0.7, these multiple carriers are defined as a list in YAML.
E.g.,

```yaml
techs:
  chp:
    name: Combined heat and power (CHP) plant
    carrier_in: gas
    carrier_out: [electricity, heat]
  ashp:
    name: Air source heat pump
    carrier_in: electricity
    carrier_out: [cooling, heat]
  slightly_cleaner_coal_plant:
    name: Dual fuel coal fired power station
    carrier_in: [coal, biofuel]
    carrier_out: electricity
  nuclear:
    name: Nuclear power station
    carrier_out: [electricity, nuclear_waste]
```

The default math applied to these multiple carriers is that inflow carrier requirements are based on the sum of the outflow carriers and vice-versa (see [the base math][balance_conversion]).
This is valid for our heat pump and coal-fired power plant examples above, but not the nuclear and CHP examples.
In these examples, the inflow is linked to a specific outflow (gas / nuclear fuel consumption is a function of electricity production).
The other carrier outflows are then linked to the "primary" outflow.

To capture this slightly different math, you will need to [apply your own math](../user_defined_math/index.md).
For example, the CHP example is dealt with in our [urban scale example model](../examples/urban_scale/index.md#interlude-user-defined-math) and in an [example additional math file][chp-plants].

No matter how you formulate your math, you can (and probably will need to) extend your technology parameters to account for these different carriers.
For instance, to differentiate `flow_cap_max` between carriers or to assign different conversion efficiencies / costs:

```yaml
techs:
  chp:
    name: Combined heat and power (CHP) plant
    carrier_in: gas
    carrier_out: [electricity, heat]
    flow_cap_max:
      data: 100
      index: electricity
      dims: carriers
    cost_flow_cap:
      data: 0.1
      index: [[electricity, monetary]]
      dims: [carriers, costs]
  ashp:
    name: Air source heat pump
    carrier_in: electricity
    carrier_out: [cooling, heat]
    flow_out_eff:
      data: [3, 4]
      index: [cooling, heat]
      dims: carriers
  slightly_cleaner_coal_plant:
    name: Dual fuel coal fired power station
    carrier_in: [coal, biofuel]
    carrier_out: electricity
    cost_flow_in:
      data: [0.1, 0.5]
      index: [[coal, monetary], [biofuel, monetary]]
      dims: [carriers, costs]
  nuclear:
    name: Nuclear power station
    carrier_out: [electricity, nuclear_waste]
    waste_per_flow_out: 0.1  # (1)!
```

1. This is a user-defined parameter that you won't find in the `Parameters` section of our [pre-defined base math documentation][base-math].
You can use it in your own math to link nuclear waste outflow with electricity outflow.

## Activating storage buffers in non-storage technologies

As their name suggests, `storage` technologies can store carriers between timesteps.
In addition, any other abstract base technology can define a storage buffer using `include_storage: true`.
This means that any inflow into the technology can be stored between timesteps and released from that technology as an outflow at a later timestep.

You may like to use this if you have a `supply` source that can be stored within the system before being used to produce a carrier (think superheated fluid in a concentrated solar power array, biomass in a biogas plant, etc.),
or if you want your stored carrier to be converted on outflow (i.e., within a `conversion` technology).

## Revenues and carrier export

It is possible to specify revenues for technologies simply by setting a negative cost value.
For example, to consider a feed-in tariff for PV generation, it could be given a negative operational cost equal to the real operational cost minus the level of feed-in tariff received.

Export is an extension of this, allowing a carrier to be removed from the system without meeting demand.
This is analogous to e.g. domestic PV technologies being able to export excess electricity to the national grid.
A cost (or negative cost: revenue) can then be applied to export.
There is an example of this in our [urban scale example model](../examples/urban_scale/index.md#revenue-by-export).

!!! note
    Negative costs can be applied to capacity costs, but the user must an ensure a capacity limit has been set.
    Otherwise, optimisation will fail due to being unbounded.

## Area use constraints

Several optional constraints can be used to specify area-related restrictions on technology use.
These constraints may reflect the fact that resources scale with area of deployed technology (think solar panels or biofuel crops).
Here you would use `source_unit: per_area` for your `supply` technology.
These constraints may also reflect limits on deployment according to available space (think limited rooftop space for solar panels or land available for biofuel crop cultivation).
Here you would use `area_use_min/max` and then link area use with technology flow capacity with `area_use_per_flow_cap`, which forces `area_use` to follow `flow_cap` with the given numerical ratio (e.g. setting to 1.5 means that `area_use == 1.5 * flow_cap`).

At a given node, you may also have a limited amount of space that technologies need to compete for.
Setting `available_area` at a node will force the combined `area_use` of all technologies at that node to a given value.

## One way transmission links

Transmission links are bidirectional by default.
To force unidirectionality for a given technology along a given link, you have to set the `one_way` constraint in the definition of that technology:

```yaml
techs:
  region1_to_region2:
    link_from: region1
    link_to: region2
    base_tech: transmission
    one_way: true
```

This will only allow transmission from `region1` to `region2`.
To swap the direction, `link_to` and `link_from` must be swapped.

## Per-distance transmission constraints

Transmission technologies can additionally specify per-distance efficiency (loss) with `flow_out_eff_per_distance` and per-distance costs with `cost_flow_cap_per_distance`:

```yaml
techs:
  my_transmission_tech:
    # "efficiency" (1-loss) per unit of distance
    flow_out_eff_per_distance: 0.99
    cost_flow_cap_per_distance:
      data: 10
      index: monetary
      dims: costs
    distance: 500
```

The distance can be specified in transmission links (as above) or, if no distance is given, from node `latitude` and `longitude` coordinates.
In the latter case, Calliope will compute distances automatically based on the length of a straight line connecting the locations, following the curvature of the earth.

!!! note
    Automatically derived distance values are provided in kilometres.
    If your model has a small geographic scope, you may prefer to use metres.
    You can update the configuration option `config.init.distance_unit` to reflect this preference.

## Cyclic storage

For any technologies with storage (`storage` technologies or those with [storage buffer](#activating-storage-buffers-in-non-storage-technologies)), it is possible to link the storage at either end of the timeseries using the `cyclic_storage` parameter.
This allows the user to better represent multiple years by just modelling one year.
Cyclic storage is activated by default (to deactivate: `cyclic_storage: false` in all storage technologies).
As a result, a technology's initial stored energy at a given location will be equal to its stored energy at the end of the model's last timestep.

For example, for a model running over a full year at hourly resolution, the initial storage at `Jan 1st 00:00` will be forced equal to the storage at the end of the timestep `Dec 31st 23:00`.
By setting `storage_initial` for a technology, it is also possible to fix the value in the last timestep.
For instance, with `cyclic_storage: true` and a `storage_initial: 0`, the stored energy *must* be zero by the end of the time horizon.

Without cyclic storage in place, technologies can have any amount of stored energy by the end of the timeseries.
This may prove useful in some cases, but makes little sense if you imagine your technologies operating in the same manner year-on-year.
Any excess stored energy would result in double the excess the following year, and so on.

!!! note
    Cyclic storage also functions when [time clustering](time.md#time-clustering), if allowing storage to be tracked between clusters.
    However, it cannot be used in [`operate` mode](../basic/modes.md#operate-mode).
