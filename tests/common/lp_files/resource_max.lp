\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(my_constraint)(0)_:
+1 variables(resource_con)(a__test_supply_plus__2005_01_01_00_00)
-24.0 variables(resource_cap)(a__test_supply_plus)
<= 0.0

c_u_constraints(my_constraint)(1)_:
-24.0 variables(resource_cap)(a__test_supply_plus)
+1 variables(resource_con)(a__test_supply_plus__2005_01_02_00_00)
<= 0.0

c_u_constraints(my_constraint)(2)_:
+1 variables(resource_con)(b__test_supply_plus__2005_01_01_00_00)
-24.0 variables(resource_cap)(b__test_supply_plus)
<= 0.0

c_u_constraints(my_constraint)(3)_:
-24.0 variables(resource_cap)(b__test_supply_plus)
+1 variables(resource_con)(b__test_supply_plus__2005_01_02_00_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(resource_con)(a__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(resource_cap)(a__test_supply_plus) <= +inf
   0 <= variables(resource_con)(a__test_supply_plus__2005_01_02_00_00) <= +inf
   0 <= variables(resource_con)(b__test_supply_plus__2005_01_01_00_00) <= +inf
   0 <= variables(resource_cap)(b__test_supply_plus) <= +inf
   0 <= variables(resource_con)(b__test_supply_plus__2005_01_02_00_00) <= +inf
end
