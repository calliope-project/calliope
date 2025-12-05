\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(storage_inter_min)(a__test_storage__2005_01_01_00_00)_:
+0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_01_00_00)
+1 variables(storage_intra_cluster_min)(a__test_storage__0)
>= 0.0

c_l_constraints(storage_inter_min)(a__test_storage__2005_01_02_00_00)_:
+0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_02_00_00)
+1 variables(storage_intra_cluster_min)(a__test_storage__1)
>= 0.0

c_l_constraints(storage_inter_min)(a__test_storage__2005_01_03_00_00)_:
+1 variables(storage_intra_cluster_min)(a__test_storage__1)
+0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_03_00_00)
>= 0.0

c_l_constraints(storage_inter_min)(a__test_storage__2005_01_04_00_00)_:
+1 variables(storage_intra_cluster_min)(a__test_storage__0)
+0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_04_00_00)
>= 0.0

c_l_constraints(storage_inter_min)(a__test_storage__2005_01_05_00_00)_:
+1 variables(storage_intra_cluster_min)(a__test_storage__0)
+0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00)
>= 0.0

c_l_constraints(storage_inter_min)(b__test_storage__2005_01_01_00_00)_:
+0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_01_00_00)
+1 variables(storage_intra_cluster_min)(b__test_storage__0)
>= 0.0

c_l_constraints(storage_inter_min)(b__test_storage__2005_01_02_00_00)_:
+0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_02_00_00)
+1 variables(storage_intra_cluster_min)(b__test_storage__1)
>= 0.0

c_l_constraints(storage_inter_min)(b__test_storage__2005_01_03_00_00)_:
+1 variables(storage_intra_cluster_min)(b__test_storage__1)
+0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_03_00_00)
>= 0.0

c_l_constraints(storage_inter_min)(b__test_storage__2005_01_04_00_00)_:
+1 variables(storage_intra_cluster_min)(b__test_storage__0)
+0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_04_00_00)
>= 0.0

c_l_constraints(storage_inter_min)(b__test_storage__2005_01_05_00_00)_:
+1 variables(storage_intra_cluster_min)(b__test_storage__0)
+0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_05_00_00)
>= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_01_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_min)(a__test_storage__0) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_02_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_min)(a__test_storage__1) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_03_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_04_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_01_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_min)(b__test_storage__0) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_02_00_00) <= +inf
   -inf <= variables(storage_intra_cluster_min)(b__test_storage__1) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_03_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_04_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_05_00_00) <= +inf
end
