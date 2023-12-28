\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(chp_backpressure_line_equals)(a__test_chp__2005_01_01_00_00)_:
+1 variables(flow_out)(a__test_chp__electricity__2005_01_01_00_00)
-1.25 variables(flow_out)(a__test_chp__heat__2005_01_01_00_00)
= 0.0

c_e_constraints(chp_backpressure_line_equals)(a__test_chp__2005_01_01_01_00)_:
+1 variables(flow_out)(a__test_chp__electricity__2005_01_01_01_00)
-1.25 variables(flow_out)(a__test_chp__heat__2005_01_01_01_00)
= 0.0

c_e_constraints(balance_conversion)(a__test_chp__2005_01_01_00_00)_:
+1.6666666666666667 variables(flow_out)(a__test_chp__electricity__2005_01_01_00_00)
-1.0 variables(flow_in)(a__test_chp__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_conversion)(a__test_chp__2005_01_01_01_00)_:
+1.6666666666666667 variables(flow_out)(a__test_chp__electricity__2005_01_01_01_00)
-1.0 variables(flow_in)(a__test_chp__gas__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_chp__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_chp__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_chp__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_chp__heat__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_chp__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_chp__gas__2005_01_01_01_00) <= +inf
end
