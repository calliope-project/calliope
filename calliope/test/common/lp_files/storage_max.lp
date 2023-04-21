\* Source Pyomo model name=None *\

min
objectives(0):
+2 ONE_VAR_CONSTANT

s.t.

c_u_constraints(storage_max)(0)_:
+1 variables(storage)(test_storage__a__2005_01_01_00_00)
-1 variables(storage_cap)(test_storage__a)
<= 0

c_u_constraints(storage_max)(1)_:
+1 variables(storage)(test_storage__a__2005_01_01_01_00)
-1 variables(storage_cap)(test_storage__a)
<= 0

c_u_constraints(storage_max)(2)_:
+1 variables(storage)(test_storage__b__2005_01_01_00_00)
-1 variables(storage_cap)(test_storage__b)
<= 0

c_u_constraints(storage_max)(3)_:
+1 variables(storage)(test_storage__b__2005_01_01_01_00)
-1 variables(storage_cap)(test_storage__b)
<= 0

c_e_ONE_VAR_CONSTANT:
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= variables(storage_cap)(test_storage__a) <= 15
   0 <= variables(storage_cap)(test_storage__b) <= 15
   0 <= variables(storage)(test_storage__a__2005_01_01_00_00) <= +inf
   0 <= variables(storage)(test_storage__a__2005_01_01_01_00) <= +inf
   0 <= variables(storage)(test_storage__b__2005_01_01_00_00) <= +inf
   0 <= variables(storage)(test_storage__b__2005_01_01_01_00) <= +inf
end
