\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(ramping_up_after_low_output)(a__test_storage__electricity__2005_01_02_00_00)_:
+0.041666666666666664 variables(flow_out)(a__test_storage__electricity__2005_01_02_00_00)
-0.041666666666666664 variables(flow_in)(a__test_storage__electricity__2005_01_02_00_00)
-1 variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00)
+1 variables(flow_in)(a__test_storage__electricity__2005_01_01_00_00)
-23.625 variables(flow_cap)(a__test_storage__electricity)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_storage__electricity__2005_01_02_00_00)_:
-0.041666666666666664 variables(flow_out)(a__test_storage__electricity__2005_01_02_00_00)
+0.041666666666666664 variables(flow_in)(a__test_storage__electricity__2005_01_02_00_00)
+1 variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00)
-1 variables(flow_in)(a__test_storage__electricity__2005_01_01_00_00)
-23.625 variables(flow_cap)(a__test_storage__electricity)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_out)(a__test_storage__electricity__2005_01_02_00_00) <= +inf
   0.0 <= variables(flow_in)(a__test_storage__electricity__2005_01_02_00_00) <= +inf
   0.0 <= variables(flow_out)(a__test_storage__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_in)(a__test_storage__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_storage__electricity) <= 10.0
end
