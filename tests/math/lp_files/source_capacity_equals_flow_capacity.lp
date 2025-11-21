\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(source_capacity_equals_flow_capacity)(a__test_supply_plus__electricity)_:
+1 variables(source_cap)(a__test_supply_plus)
-1 variables(flow_cap)(a__test_supply_plus__electricity)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(source_cap)(a__test_supply_plus) <= +inf
   0.0 <= variables(flow_cap)(a__test_supply_plus__electricity) <= 15.0
end
