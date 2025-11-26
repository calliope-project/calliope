\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(area_use_minimum)(a__test_supply_elec)_:
+1 variables(area_use)(a__test_supply_elec)
>= 5.0

c_l_constraints(area_use_minimum)(b__test_supply_elec)_:
+1 variables(area_use)(b__test_supply_elec)
>= 5.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(area_use)(a__test_supply_elec) <= +inf
   0 <= variables(area_use)(b__test_supply_elec) <= +inf
end
