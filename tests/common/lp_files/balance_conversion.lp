\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(balance_conversion)(a__test_conversion__2005_01_01_00_00)_:
+1.1111111111111112 variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00)
-1.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_conversion)(a__test_conversion__2005_01_01_01_00)_:
+1.1111111111111112 variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00)
-1.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
= 0.0

c_e_constraints(balance_conversion)(b__test_conversion__2005_01_01_00_00)_:
+1.1111111111111112 variables(flow_out)(b__test_conversion__heat__2005_01_01_00_00)
-1.0 variables(flow_in)(b__test_conversion__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_conversion)(b__test_conversion__2005_01_01_01_00)_:
+1.1111111111111112 variables(flow_out)(b__test_conversion__heat__2005_01_01_01_00)
-1.0 variables(flow_in)(b__test_conversion__gas__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_conversion__heat__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion__gas__2005_01_01_01_00) <= +inf
end
