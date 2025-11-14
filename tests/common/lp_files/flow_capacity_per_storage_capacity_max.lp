\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(flow_capacity_per_storage_capacity_max)(a__test_storage__electricity)_:
+1 variables(flow_cap)(a__test_storage__electricity)
-2.0 variables(storage_cap)(a__test_storage)
<= 0.0

c_u_constraints(flow_capacity_per_storage_capacity_max)(b__test_storage__electricity)_:
+1 variables(flow_cap)(b__test_storage__electricity)
-0.5 variables(storage_cap)(b__test_storage)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_cap)(a__test_storage__electricity) <= 10.0
   0.0 <= variables(storage_cap)(a__test_storage) <= 15.0
   0.0 <= variables(flow_cap)(b__test_storage__electricity) <= 10.0
   0.0 <= variables(storage_cap)(b__test_storage) <= 15.0
end
