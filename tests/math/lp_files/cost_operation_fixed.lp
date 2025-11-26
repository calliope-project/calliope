\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+0.01141552511415525 variables(flow_cap)(a__test_supply_elec__electricity)
+0.01141552511415525 variables(flow_cap)(b__test_supply_elec__electricity)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0.0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 10.0
end
