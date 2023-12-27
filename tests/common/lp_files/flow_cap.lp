\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+1 variables(flow_cap)(a__test_supply_elec__electricity)
+1 variables(flow_cap)(b__test_supply_elec__electricity)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   1.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= +inf
   0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 100.0
end
