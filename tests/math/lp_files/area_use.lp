\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+1 variables(area_use)(a__test_supply_elec)
+1 variables(area_use)(b__test_supply_elec)

s.t.

c_l_constraints(area_use_minimum)(a__test_supply_elec)_:
+1 variables(area_use)(a__test_supply_elec)
>= 1.0

bounds
   0 <= variables(area_use)(a__test_supply_elec) <= +inf
   0 <= variables(area_use)(b__test_supply_elec) <= 100.0
end
