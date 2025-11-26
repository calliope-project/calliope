\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(balance_storage)(a__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(a__test_storage__2005_01_01_00_00)
+1.1111111111111112 variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00)
-1 variables(flow_in)(a__test_storage__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_storage)(a__test_storage__2005_01_03_00_00)_:
+1 variables(storage)(a__test_storage__2005_01_03_00_00)
+1.1111111111111112 variables(flow_out)(a__test_storage__electricity__2005_01_03_00_00)
-1 variables(flow_in)(a__test_storage__electricity__2005_01_03_00_00)
= 0.0

c_e_constraints(balance_storage)(b__test_storage__2005_01_01_00_00)_:
+1 variables(storage)(b__test_storage__2005_01_01_00_00)
+1.1111111111111112 variables(flow_out)(b__test_storage__electricity__2005_01_01_00_00)
-1 variables(flow_in)(b__test_storage__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_storage)(b__test_storage__2005_01_03_00_00)_:
+1 variables(storage)(b__test_storage__2005_01_03_00_00)
+1.1111111111111112 variables(flow_out)(b__test_storage__electricity__2005_01_03_00_00)
-1 variables(flow_in)(b__test_storage__electricity__2005_01_03_00_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   -inf <= variables(storage)(a__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_storage__electricity__2005_01_01_00_00) <= +inf
   -inf <= variables(storage)(a__test_storage__2005_01_03_00_00) <= +inf
   0 <= variables(flow_out)(a__test_storage__electricity__2005_01_03_00_00) <= +inf
   0 <= variables(flow_in)(a__test_storage__electricity__2005_01_03_00_00) <= +inf
   -inf <= variables(storage)(b__test_storage__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_storage__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_storage__electricity__2005_01_01_00_00) <= +inf
   -inf <= variables(storage)(b__test_storage__2005_01_03_00_00) <= +inf
   0 <= variables(flow_out)(b__test_storage__electricity__2005_01_03_00_00) <= +inf
   0 <= variables(flow_in)(b__test_storage__electricity__2005_01_03_00_00) <= +inf
end
