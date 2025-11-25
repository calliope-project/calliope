\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(source_capacity_minimum)(a__test_supply_plus)_:
+1 variables(source_cap)(a__test_supply_plus)
>= 30.0

c_l_constraints(source_capacity_minimum)(b__test_supply_plus)_:
+1 variables(source_cap)(b__test_supply_plus)
>= 30.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(source_cap)(a__test_supply_plus) <= +inf
   0 <= variables(source_cap)(b__test_supply_plus) <= +inf
end
