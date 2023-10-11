\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(max_time_varying_flow_cap)(0)_:
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.8 variables(flow_cap)(a__test_supply_elec)
<= 0

c_u_constraints(max_time_varying_flow_cap)(1)_:
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
-0.5 variables(flow_cap)(a__test_supply_elec)
<= 0

c_u_constraints(max_time_varying_flow_cap)(2)_:
+1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
-0.8 variables(flow_cap)(b__test_supply_elec)
<= 0

c_u_constraints(max_time_varying_flow_cap)(3)_:
+1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
-0.5 variables(flow_cap)(b__test_supply_elec)
<= 0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_cap)(a__test_supply_elec) <= +inf
   0 <= variables(flow_cap)(b__test_supply_elec) <= +inf
end
