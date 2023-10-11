\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(max_time_varying_storage)(0)_:
+1 variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00)
-0.8 variables(storage)(a__test_storage__2005_01_01_00_00)
<= 0

c_u_constraints(max_time_varying_storage)(1)_:
+1 variables(flow_out)(a__test_storage__electricity__2005_01_01_01_00)
-0.5 variables(storage)(a__test_storage__2005_01_01_01_00)
<= 0

c_u_constraints(max_time_varying_storage)(2)_:
+1 variables(flow_out)(b__test_storage__electricity__2005_01_01_00_00)
-0.8 variables(storage)(b__test_storage__2005_01_01_00_00)
<= 0

c_u_constraints(max_time_varying_storage)(3)_:
+1 variables(flow_out)(b__test_storage__electricity__2005_01_01_01_00)
-0.5 variables(storage)(b__test_storage__2005_01_01_01_00)
<= 0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_storage__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_storage__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_storage__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(storage)(a__test_storage__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(storage)(a__test_storage__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(storage)(b__test_storage__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(storage)(b__test_storage__electricity__2005_01_01_01_00) <= +inf
end
