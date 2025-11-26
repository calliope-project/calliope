\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+50.0 variables(storage_cap)(a__test_storage)
+50.0 variables(storage_cap)(b__test_storage)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(storage_cap)(a__test_storage) <= 15.0
   0.0 <= variables(storage_cap)(b__test_storage) <= 15.0
end
