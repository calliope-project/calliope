\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(area_use_per_flow_capacity)(a__test_supply_elec__electricity)_:
+1 variables(area_use)(a__test_supply_elec)
-1.5 variables(flow_cap)(a__test_supply_elec__electricity)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(area_use)(a__test_supply_elec) <= +inf
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
end
