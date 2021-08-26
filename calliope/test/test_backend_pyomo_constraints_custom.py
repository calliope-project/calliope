"""
custom_constraints:
    parameters:
        carrier_prod_share: 0.25
        net_import_share: 0.7
        energy_cap_share: 0.2
        cost_cap: 1e6
        cost_var_cap: 1e4
        cost_investment_cap: 1e3
        resource_area_cap: 100
    constraints:
        energy_cap_share:
            sets:
                lhs:
                    techs: [onshore_wind]
                    locs:
                rhs:
                    techs: [onshore_wind, offshore_wind, gas, coal]
            iterate_over:
                locs: __ALL__
                carriers: [electricity]
            eq:
                lhs: sum(energy_cap[lhs])
                operator: <=
                rhs: sum(energy_cap[rhs])
"""
