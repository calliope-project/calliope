\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(supply_share_per_timestep_equals_per_tech)(a__test_supply_elec__2005_01_01_00_00)_:
+0.5 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.5 variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_00_00)
-0.5 variables(flow_out)(a__test_supply_plus__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(supply_share_per_timestep_equals_per_tech)(a__test_supply_elec__2005_01_01_01_00)_:
+0.5 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
-0.5 variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_01_00)
-0.5 variables(flow_out)(a__test_supply_plus__electricity__2005_01_01_01_00)
= 0.0

c_e_constraints(supply_share_per_timestep_equals_per_tech)(b__test_supply_elec__2005_01_01_00_00)_:
+0.19999999999999996 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
-0.8 variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_00_00)
-0.8 variables(flow_out)(b__test_supply_plus__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(supply_share_per_timestep_equals_per_tech)(b__test_supply_elec__2005_01_01_01_00)_:
+0.19999999999999996 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
-0.8 variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_01_00)
-0.8 variables(flow_out)(b__test_supply_plus__electricity__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_plus__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_plus__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_plus__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_plus__electricity__2005_01_01_01_00) <= +inf
end
