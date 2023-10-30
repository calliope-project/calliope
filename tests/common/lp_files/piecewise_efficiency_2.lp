














   0 <= variables(available_flow_cap)(a__test_conversion__2005_01_01_00_00) <= 15.0
   0 <= variables(available_flow_cap)(a__test_conversion__2005_01_01_01_00) <= 15.0
   0 <= variables(flow_cap)(a__test_conversion) <= 15.0
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
   0 <= variables(operating_units)(a__test_conversion__2005_01_01_00_00) <= +inf
   0 <= variables(operating_units)(a__test_conversion__2005_01_01_01_00) <= +inf
   1 <= ONE_VAR_CONSTANT <= 1
  variables(operating_units)(a__test_conversion__2005_01_01_00_00)
  variables(operating_units)(a__test_conversion__2005_01_01_01_00)
+1 variables(available_flow_cap)(a__test_conversion__2005_01_01_00_00)
+1 variables(available_flow_cap)(a__test_conversion__2005_01_01_00_00)
+1 variables(available_flow_cap)(a__test_conversion__2005_01_01_01_00)
+1 variables(available_flow_cap)(a__test_conversion__2005_01_01_01_00)
+1 variables(flow_cap)(a__test_conversion)
+1 variables(flow_cap)(a__test_conversion)
+14 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
+14 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
+15.0 variables(operating_units)(a__test_conversion__2005_01_01_00_00)
+15.0 variables(operating_units)(a__test_conversion__2005_01_01_01_00)
+2.0 ONE_VAR_CONSTANT
+5 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
+5 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
+7 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
+7 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
-1 variables(available_flow_cap)(a__test_conversion__2005_01_01_00_00)
-1 variables(available_flow_cap)(a__test_conversion__2005_01_01_01_00)
-1 variables(flow_cap)(a__test_conversion)
-1 variables(flow_cap)(a__test_conversion)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
-15.0 variables(operating_units)(a__test_conversion__2005_01_01_00_00)
-15.0 variables(operating_units)(a__test_conversion__2005_01_01_01_00)
-16 variables(available_flow_cap)(a__test_conversion__2005_01_01_00_00)
-16 variables(available_flow_cap)(a__test_conversion__2005_01_01_01_00)
-2 variables(available_flow_cap)(a__test_conversion__2005_01_01_00_00)
-2 variables(available_flow_cap)(a__test_conversion__2005_01_01_01_00)
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 0.0
<= 15.0
<= 15.0
\* Source Pyomo model name=None *\
bounds
c_u_constraints(available_flow_cap_binary)(0)_:
c_u_constraints(available_flow_cap_binary)(1)_:
c_u_constraints(available_flow_cap_binary_continuous_switch)(0)_:
c_u_constraints(available_flow_cap_binary_continuous_switch)(1)_:
c_u_constraints(available_flow_cap_continuous)(0)_:
c_u_constraints(available_flow_cap_continuous)(1)_:
c_u_constraints(piecewise_efficiency)(0)_:
c_u_constraints(piecewise_efficiency)(1)_:
c_u_constraints(piecewise_efficiency)(2)_:
c_u_constraints(piecewise_efficiency)(3)_:
c_u_constraints(piecewise_efficiency)(4)_:
c_u_constraints(piecewise_efficiency)(5)_:
end
general
min
objectives(dummy_obj)(0):
s.t.
