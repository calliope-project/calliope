\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+30.0 variables(source_cap)(a__test_supply_plus)
+30.0 variables(source_cap)(b__test_supply_plus)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(source_cap)(a__test_supply_plus) <= +inf
   0.0 <= variables(source_cap)(b__test_supply_plus) <= +inf
end
