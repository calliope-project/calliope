\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(balance_storage_inter)(b__test_storage__2005_01_01_00_00)_:
+1 variables(storage_inter_cluster)(b__test_storage__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(a__test_storage__2005_01_01_00_00)_:
+1 variables(storage_inter_cluster)(a__test_storage__2005_01_01_00_00)
-0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00)
-1 variables(storage)(a__test_storage__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(a__test_storage__2005_01_02_00_00)_:
-0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_01_00_00)
-1 variables(storage)(a__test_storage__2005_01_01_00_00)
+1 variables(storage_inter_cluster)(a__test_storage__2005_01_02_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(a__test_storage__2005_01_03_00_00)_:
-0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_02_00_00)
+1 variables(storage_inter_cluster)(a__test_storage__2005_01_03_00_00)
-1 variables(storage)(a__test_storage__2005_01_03_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(a__test_storage__2005_01_04_00_00)_:
-0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_03_00_00)
-1 variables(storage)(a__test_storage__2005_01_03_00_00)
+1 variables(storage_inter_cluster)(a__test_storage__2005_01_04_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(a__test_storage__2005_01_05_00_00)_:
+1 variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00)
-1 variables(storage)(a__test_storage__2005_01_01_00_00)
-0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_04_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(b__test_storage__2005_01_02_00_00)_:
-0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_01_00_00)
+1 variables(storage_inter_cluster)(b__test_storage__2005_01_02_00_00)
-1 variables(storage)(b__test_storage__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(b__test_storage__2005_01_03_00_00)_:
-0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_02_00_00)
+1 variables(storage_inter_cluster)(b__test_storage__2005_01_03_00_00)
-1 variables(storage)(b__test_storage__2005_01_03_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(b__test_storage__2005_01_04_00_00)_:
-0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_03_00_00)
-1 variables(storage)(b__test_storage__2005_01_03_00_00)
+1 variables(storage_inter_cluster)(b__test_storage__2005_01_04_00_00)
= 0.0

c_e_constraints(balance_storage_inter)(b__test_storage__2005_01_05_00_00)_:
-1 variables(storage)(b__test_storage__2005_01_01_00_00)
-0.7856781408072188 variables(storage_inter_cluster)(b__test_storage__2005_01_04_00_00)
+1 variables(storage_inter_cluster)(b__test_storage__2005_01_05_00_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00) <= +inf
   -inf <= variables(storage)(a__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_02_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_03_00_00) <= +inf
   -inf <= variables(storage)(a__test_storage__2005_01_03_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_04_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_02_00_00) <= +inf
   -inf <= variables(storage)(b__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_03_00_00) <= +inf
   -inf <= variables(storage)(b__test_storage__2005_01_03_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_04_00_00) <= +inf
   0 <= variables(storage_inter_cluster)(b__test_storage__2005_01_05_00_00) <= +inf
end
