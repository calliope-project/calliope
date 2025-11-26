\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(source_capacity_minimum)(a__test_supply_plus)_:
+30.0 variables(purchased_units)(a__test_supply_plus)
-1 variables(source_cap)(a__test_supply_plus)
<= 0.0

c_u_constraints(source_capacity_minimum)(b__test_supply_plus)_:
+30.0 variables(purchased_units)(b__test_supply_plus)
-1 variables(source_cap)(b__test_supply_plus)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(purchased_units)(a__test_supply_plus) <= +inf
   0 <= variables(source_cap)(a__test_supply_plus) <= +inf
   0.0 <= variables(purchased_units)(b__test_supply_plus) <= +inf
   0 <= variables(source_cap)(b__test_supply_plus) <= +inf
general
  variables(purchased_units)(a__test_supply_plus)
  variables(purchased_units)(b__test_supply_plus)
end
