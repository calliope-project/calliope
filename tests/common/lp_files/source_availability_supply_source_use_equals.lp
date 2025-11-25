\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(source_availability_supply)(a__test_supply_plus__2005_01_01_00_00)_:
+1 variables(source_use)(a__test_supply_plus__2005_01_01_00_00)
= 5.0

c_e_constraints(source_availability_supply)(a__test_supply_plus__2005_01_02_00_00)_:
+1 variables(source_use)(a__test_supply_plus__2005_01_02_00_00)
= 5.0

c_u_constraints(source_availability_supply)(b__test_supply_plus__2005_01_01_00_00)_:
+1 variables(source_use)(b__test_supply_plus__2005_01_01_00_00)
<= 80.0

c_u_constraints(source_availability_supply)(b__test_supply_plus__2005_01_02_00_00)_:
+1 variables(source_use)(b__test_supply_plus__2005_01_02_00_00)
<= 80.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(source_use)(a__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(a__test_supply_plus__2005_01_02_00_00) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_02_00_00) <= +inf
end
