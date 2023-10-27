\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(my_constraint)(a__test_supply_plus__2005_01_01_00_00)_:
+1 variables(source_use)(a__test_supply_plus__2005_01_01_00_00)
-24.0 variables(source_cap)(a__test_supply_plus)
<= 0.0

c_u_constraints(my_constraint)(a__test_supply_plus__2005_01_02_00_00)_:
-24.0 variables(source_cap)(a__test_supply_plus)
+1 variables(source_use)(a__test_supply_plus__2005_01_02_00_00)
<= 0.0

c_u_constraints(my_constraint)(b__test_supply_plus__2005_01_01_00_00)_:
+1 variables(source_use)(b__test_supply_plus__2005_01_01_00_00)
-24.0 variables(source_cap)(b__test_supply_plus)
<= 0.0

c_u_constraints(my_constraint)(b__test_supply_plus__2005_01_02_00_00)_:
-24.0 variables(source_cap)(b__test_supply_plus)
+1 variables(source_use)(b__test_supply_plus__2005_01_02_00_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(source_use)(a__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(source_cap)(a__test_supply_plus) <= +inf
   0 <= variables(source_use)(a__test_supply_plus__2005_01_02_00_00) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(source_cap)(b__test_supply_plus) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_02_00_00) <= +inf
end
