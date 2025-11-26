\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(storage_discharge_depth_limit)(a__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(a__test_storage__2005_01_01_00_00)
-0.2 variables(storage_cap)(a__test_storage)
>= 0.0

c_l_constraints(storage_discharge_depth_limit)(a__test_storage__2005_01_01_01_00)_:
-0.2 variables(storage_cap)(a__test_storage)
+1 variables(storage)(a__test_storage__2005_01_01_01_00)
>= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(storage)(a__test_storage__2005_01_01_00_00) <= +inf
   0.0 <= variables(storage_cap)(a__test_storage) <= 15.0
   0 <= variables(storage)(a__test_storage__2005_01_01_01_00) <= +inf
end
