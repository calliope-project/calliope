\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(storage_capacity_minimum)(a__test_supply_elec)_:
+1 variables(storage_cap)(a__test_supply_elec)
>= 20.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(storage_cap)(a__test_supply_elec) <= +inf
end
