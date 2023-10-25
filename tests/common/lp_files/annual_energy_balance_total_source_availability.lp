\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(annual_energy_balance_total_source_availability)(test_supply_plus)_:
+1 variables(source_use)(a__test_supply_plus__2005_01_01_00_00)
+1 variables(source_use)(a__test_supply_plus__2005_01_01_01_00)
+1 variables(source_use)(b__test_supply_plus__2005_01_01_00_00)
+1 variables(source_use)(b__test_supply_plus__2005_01_01_01_00)
<= 10.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(source_use)(a__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(a__test_supply_plus__2005_01_01_01_00) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_01_01_00) <= +inf
end
