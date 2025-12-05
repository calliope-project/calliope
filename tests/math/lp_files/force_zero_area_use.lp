\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(force_zero_area_use)(a__test_supply_elec)_:
+1 variables(area_use)(a__test_supply_elec)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(area_use)(a__test_supply_elec) <= 100.0
end
