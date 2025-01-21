\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+1 variables(source_cap)(a__test_supply_elec)
+1 variables(source_cap)(b__test_supply_elec)

s.t.

c_l_constraints(source_capacity_minimum)(a__test_supply_elec)_:
+1 variables(source_cap)(a__test_supply_elec)
>= 1.0

bounds
   0 <= variables(source_cap)(a__test_supply_elec) <= +inf
   0 <= variables(source_cap)(b__test_supply_elec) <= 100.0
end
