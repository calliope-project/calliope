\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(demand_share_per_timestep_decision_main_min)(0)_:
+4.95 variables(demand_share_per_timestep_decision)(a__test_conversion_plus)
-1 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_min)(1)_:
+4.95 variables(demand_share_per_timestep_decision)(a__test_conversion_plus)
-1 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_min)(2)_:
+4.95 variables(demand_share_per_timestep_decision)(a__test_supply_elec)
-1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_min)(3)_:
+4.95 variables(demand_share_per_timestep_decision)(a__test_supply_elec)
-1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_max)(0)_:
-5.05 variables(demand_share_per_timestep_decision)(a__test_conversion_plus)
+1 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_max)(1)_:
-5.05 variables(demand_share_per_timestep_decision)(a__test_conversion_plus)
+1 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_max)(2)_:
-5.05 variables(demand_share_per_timestep_decision)(a__test_supply_elec)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(demand_share_per_timestep_decision_main_max)(3)_:
-5.05 variables(demand_share_per_timestep_decision)(a__test_supply_elec)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(demand_share_per_timestep_decision)(a__test_conversion_plus) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(demand_share_per_timestep_decision)(a__test_supply_elec) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
end
