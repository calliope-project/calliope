\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(flow_capacity_minimum)(a__test_supply_elec__electricity)_:
+1 variables(flow_cap)(a__test_supply_elec__electricity)
>= 10.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
end
