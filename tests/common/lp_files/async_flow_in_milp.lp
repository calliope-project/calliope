\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(async_flow_in_milp)(a__test_conversion_plus__2005_01_01_00_00)_:
+1 variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_00_00)
+1000.0 variables(async_flow_switch)(a__test_conversion_plus__2005_01_01_00_00)
<= 1000.0

c_u_constraints(async_flow_in_milp)(a__test_conversion_plus__2005_01_01_01_00)_:
+1 variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_01_00)
+1000.0 variables(async_flow_switch)(a__test_conversion_plus__2005_01_01_01_00)
<= 1000.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_00_00) <= +inf
   0 <= variables(async_flow_switch)(a__test_conversion_plus__2005_01_01_00_00) <= 1
   0 <= variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_01_00) <= +inf
   0 <= variables(async_flow_switch)(a__test_conversion_plus__2005_01_01_01_00) <= 1
binary
  variables(async_flow_switch)(a__test_conversion_plus__2005_01_01_00_00)
  variables(async_flow_switch)(a__test_conversion_plus__2005_01_01_01_00)
end
