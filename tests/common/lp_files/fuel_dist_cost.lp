\* Source Pyomo model name=None *\

min
objectives(minmax_cost_optimisation)(0):
+0.0002515252789726502 variables(flow_out_cap)(a__test_conversion_plus)
+0.0002515252789726502 variables(flow_out_cap)(a__test_supply_coal)
+0.0002515252789726502 variables(flow_out_cap)(a__test_supply_gas)
+0.0002515252789726502 variables(flow_out_cap)(b__test_conversion_plus)
+0.0002515252789726502 variables(flow_out_cap)(b__test_supply_coal)
+0.0002515252789726502 variables(flow_out_cap)(b__test_supply_gas)
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
   0 <= variables(flow_out_cap)(a__test_conversion_plus) <= 20.0
   0 <= variables(flow_out_cap)(a__test_supply_coal) <= 10.0
   0 <= variables(flow_out_cap)(a__test_supply_gas) <= 15.0
   0 <= variables(flow_out_cap)(b__test_conversion_plus) <= 20.0
   0 <= variables(flow_out_cap)(b__test_supply_coal) <= 10.0
   0 <= variables(flow_out_cap)(b__test_supply_gas) <= 15.0
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_01_00) <= +inf
end
