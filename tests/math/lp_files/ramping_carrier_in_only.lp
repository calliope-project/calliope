\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(ramping_up)(a__test_conversion__heat__2005_01_01_01_00)_:
+1.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
-1.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_conversion__heat)
<= 0.0

c_u_constraints(ramping_up)(a__test_conversion__gas__2005_01_01_01_00)_:
+1.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
-1.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_conversion__gas)
<= 0.0

c_u_constraints(ramping_down)(a__test_conversion__heat__2005_01_01_01_00)_:
-1.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
+1.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_conversion__heat)
<= 0.0

c_u_constraints(ramping_down)(a__test_conversion__gas__2005_01_01_01_00)_:
-1.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
+1.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_conversion__gas)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_conversion__heat) <= 15.0
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_conversion__gas) <= +inf
end
