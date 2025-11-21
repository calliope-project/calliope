\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(net_import_share_max)(a__2005_01_01_00_00)_:
+0.5 variables(flow_out)(a__links_a_c_elec__electricity__2005_01_01_00_00)
+0.5 variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_00_00)
-1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
+1 variables(flow_in)(a__links_a_c_elec__electricity__2005_01_01_00_00)
+1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00)
+1 variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(net_import_share_max)(a__2005_01_01_01_00)_:
+0.5 variables(flow_out)(a__links_a_c_elec__electricity__2005_01_01_01_00)
+0.5 variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_01_00)
-1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
+1 variables(flow_in)(a__links_a_c_elec__electricity__2005_01_01_01_00)
+1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00)
+1 variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(net_import_share_max)(b__2005_01_01_00_00)_:
+0.5 variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_00_00)
-1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
+1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00)
+1 variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(net_import_share_max)(b__2005_01_01_01_00)_:
+0.5 variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_01_00)
-1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
+1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00)
+1 variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(net_import_share_max)(c__2005_01_01_00_00)_:
+0.5 variables(flow_out)(c__links_a_c_elec__electricity__2005_01_01_00_00)
+1 variables(flow_in)(c__links_a_c_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(net_import_share_max)(c__2005_01_01_01_00)_:
+0.5 variables(flow_out)(c__links_a_c_elec__electricity__2005_01_01_01_00)
+1 variables(flow_in)(c__links_a_c_elec__electricity__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__links_a_c_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__links_a_c_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__links_a_c_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__links_a_c_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(c__links_a_c_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(c__links_a_c_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(c__links_a_c_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(c__links_a_c_elec__electricity__2005_01_01_01_00) <= +inf
end
