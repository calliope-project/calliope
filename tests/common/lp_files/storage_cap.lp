\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+1 variables(storage_cap)(a__test_supply_elec)
+1 variables(storage_cap)(b__test_supply_elec)

s.t.

c_l_constraints(storage_capacity_minimum)(a__test_supply_elec)_:
+1 variables(storage_cap)(a__test_supply_elec)
>= 1.0

bounds
   0 <= variables(storage_cap)(a__test_supply_elec) <= +inf
   0 <= variables(storage_cap)(b__test_supply_elec) <= 100.0
end
