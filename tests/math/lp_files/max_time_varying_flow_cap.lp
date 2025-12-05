\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(max_time_varying_flow_cap)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.8 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(max_time_varying_flow_cap)(a__test_supply_elec__electricity__2005_01_01_01_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(max_time_varying_flow_cap)(b__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
-0.8 variables(flow_cap)(b__test_supply_elec__electricity)
<= 0.0

c_u_constraints(max_time_varying_flow_cap)(b__test_supply_elec__electricity__2005_01_01_01_00)_:
-0.5 variables(flow_cap)(b__test_supply_elec__electricity)
+1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 10.0
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
end
