\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(set_storage_initial)(a__test_storage)_:
+0.7856781408072188 variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00)
-0.5 variables(storage_cap)(a__test_storage)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(storage_inter_cluster)(a__test_storage__2005_01_05_00_00) <= +inf
   0.0 <= variables(storage_cap)(a__test_storage) <= 15.0
end
