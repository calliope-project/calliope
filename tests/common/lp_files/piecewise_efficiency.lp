\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(piecewise_efficiency)(a__test_conversion__2005_01_01_00_00__0)_:
+5.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
<= 0.0

c_u_constraints(piecewise_efficiency)(a__test_conversion__2005_01_01_00_00__1)_:
+7.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-2.0 variables(available_flow_cap)(a__test_conversion__heat__2005_01_01_00_00)
<= 0.0

c_u_constraints(piecewise_efficiency)(a__test_conversion__2005_01_01_00_00__2)_:
+14.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-16.0 variables(available_flow_cap)(a__test_conversion__heat__2005_01_01_00_00)
<= 0.0

c_u_constraints(piecewise_efficiency)(a__test_conversion__2005_01_01_01_00__0)_:
+5.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
<= 0.0

c_u_constraints(piecewise_efficiency)(a__test_conversion__2005_01_01_01_00__1)_:
+7.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
-2.0 variables(available_flow_cap)(a__test_conversion__heat__2005_01_01_01_00)
<= 0.0

c_u_constraints(piecewise_efficiency)(a__test_conversion__2005_01_01_01_00__2)_:
+14.0 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
-16.0 variables(available_flow_cap)(a__test_conversion__heat__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(available_flow_cap)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   0 <= variables(available_flow_cap)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
end