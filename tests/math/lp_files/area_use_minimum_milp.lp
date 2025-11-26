\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(area_use_minimum)(a__test_supply_elec)_:
+5.0 variables(purchased_units)(a__test_supply_elec)
-1 variables(area_use)(a__test_supply_elec)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(purchased_units)(a__test_supply_elec) <= 1.0
   0 <= variables(area_use)(a__test_supply_elec) <= +inf
binary
  variables(purchased_units)(a__test_supply_elec)
end
