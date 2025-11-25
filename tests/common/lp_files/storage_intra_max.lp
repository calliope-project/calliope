\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(storage_intra_max)(a__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(a__test_storage__2005_01_01_00_00)
-1 variables(storage_intra_cluster_max)(a__test_storage__0)
<= 0.0

c_u_constraints(storage_intra_max)(a__test_storage__2005_01_03_00_00)_:
+1 variables(storage)(a__test_storage__2005_01_03_00_00)
-1 variables(storage_intra_cluster_max)(a__test_storage__1)
<= 0.0

c_u_constraints(storage_intra_max)(b__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(b__test_storage__2005_01_01_00_00)
-1 variables(storage_intra_cluster_max)(b__test_storage__0)
<= 0.0

c_u_constraints(storage_intra_max)(b__test_storage__2005_01_03_00_00)_:
+1 variables(storage)(b__test_storage__2005_01_03_00_00)
-1 variables(storage_intra_cluster_max)(b__test_storage__1)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   -inf <= variables(storage)(a__test_storage__2005_01_01_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_max)(a__test_storage__0) <= +inf
   -inf <= variables(storage)(a__test_storage__2005_01_03_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_max)(a__test_storage__1) <= +inf
   -inf <= variables(storage)(b__test_storage__2005_01_01_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_max)(b__test_storage__0) <= +inf
   -inf <= variables(storage)(b__test_storage__2005_01_03_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_max)(b__test_storage__1) <= +inf
end
