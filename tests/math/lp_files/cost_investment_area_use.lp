\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+100.0 variables(area_use)(a__test_supply_elec)
+100.0 variables(area_use)(b__test_supply_elec)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(area_use)(a__test_supply_elec) <= +inf
   0.0 <= variables(area_use)(b__test_supply_elec) <= +inf
end
