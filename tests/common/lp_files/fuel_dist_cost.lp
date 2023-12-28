\* Source Pyomo model name=None *\

min
objectives(min_cost_optimisation)(0):
+1.0 variables(flow_in)(a__test_conversion__coal__2005_01_01_00_00)
+2.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
+1.0 variables(flow_in)(a__test_conversion__coal__2005_01_01_01_00)
+2.0 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
+5.0 variables(fuel_distributor)(a__coal__2005_01_01_00_00)
+5.0 variables(fuel_distributor)(a__coal__2005_01_01_01_00)
+5.0 variables(fuel_distributor)(b__coal__2005_01_01_00_00)
+5.0 variables(fuel_distributor)(b__coal__2005_01_01_01_00)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_in)(a__test_conversion__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_01_00) <= +inf
end
