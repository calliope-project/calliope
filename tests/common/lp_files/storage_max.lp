\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(storage_max)(a__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(a__test_storage__2005_01_01_00_00)
-1 variables(storage_cap)(a__test_storage)
<= 0.0

c_u_constraints(storage_max)(a__test_storage__2005_01_01_01_00)_:
-1 variables(storage_cap)(a__test_storage)
+1 variables(storage)(a__test_storage__2005_01_01_01_00)
<= 0.0

c_u_constraints(storage_max)(b__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(b__test_storage__2005_01_01_00_00)
-1 variables(storage_cap)(b__test_storage)
<= 0.0

c_u_constraints(storage_max)(b__test_storage__2005_01_01_01_00)_:
-1 variables(storage_cap)(b__test_storage)
+1 variables(storage)(b__test_storage__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(storage)(a__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(storage_cap)(a__test_storage) <= 15.0
   0 <= variables(storage)(a__test_storage__2005_01_01_01_00) <= +inf
   0 <= variables(storage)(b__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(storage_cap)(b__test_storage) <= 15.0
   0 <= variables(storage)(b__test_storage__2005_01_01_01_00) <= +inf
end
