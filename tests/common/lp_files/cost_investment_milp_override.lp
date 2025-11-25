\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+10.0 variables(flow_cap)(a__test_supply_elec__electricity)
+1000.0 variables(purchased_units)(a__test_supply_elec)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= +inf
   0.0 <= variables(purchased_units)(a__test_supply_elec) <= 1.0
binary
  variables(purchased_units)(a__test_supply_elec)
end
