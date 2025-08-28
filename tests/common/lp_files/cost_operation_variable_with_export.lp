\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+0.1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
+0.1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
+1.0 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00)
+2.0 variables(flow_out)(a__test_conversion_plus__heat__2005_01_01_00_00)
+3.0 variables(flow_export)(a__test_conversion_plus__heat__2005_01_01_00_00)
+4.0 variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_00_00)
+1.0 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00)
+2.0 variables(flow_out)(a__test_conversion_plus__heat__2005_01_01_01_00)
+3.0 variables(flow_export)(a__test_conversion_plus__heat__2005_01_01_01_00)
+4.0 variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_01_00)
+0.1 variables(source_use)(a__test_supply_coal__2005_01_01_00_00)
+0.1 variables(source_use)(a__test_supply_coal__2005_01_01_01_00)
+5.0 variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_00_00)
+0.1 variables(source_use)(a__test_supply_elec__2005_01_01_00_00)
+5.0 variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_01_00)
+0.1 variables(source_use)(a__test_supply_elec__2005_01_01_01_00)
+0.1 variables(source_use)(a__test_supply_gas__2005_01_01_00_00)
+0.1 variables(source_use)(a__test_supply_gas__2005_01_01_01_00)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__heat__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_export)(a__test_conversion_plus__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__heat__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_export)(a__test_conversion_plus__heat__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_01_00) <= +inf
   0 <= variables(source_use)(a__test_supply_coal__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(a__test_supply_coal__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(a__test_supply_elec__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(source_use)(a__test_supply_elec__2005_01_01_01_00) <= +inf
   0 <= variables(source_use)(a__test_supply_gas__2005_01_01_00_00) <= +inf
   0 <= variables(source_use)(a__test_supply_gas__2005_01_01_01_00) <= +inf
end
