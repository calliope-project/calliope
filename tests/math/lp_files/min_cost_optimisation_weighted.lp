\* Source Pyomo model name=None *\

min
objectives(min_cost_optimisation)(0):
+2.1 variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_00_00)
+2.1 variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_01_00)
+0.023972602739726026 variables(flow_cap)(a__test_link_a_b_elec__electricity)
+2.1 variables(flow_in)(a__test_link_a_b_heat__heat__2005_01_01_00_00)
+2.1 variables(flow_in)(a__test_link_a_b_heat__heat__2005_01_01_01_00)
+0.023972602739726026 variables(flow_cap)(a__test_link_a_b_heat__heat)
+0.04794520547945205 variables(flow_cap)(a__test_supply_elec__electricity)
+2.1 variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_00_00)
+2.1 variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_01_00)
+0.023972602739726026 variables(flow_cap)(b__test_link_a_b_elec__electricity)
+2.1 variables(flow_in)(b__test_link_a_b_heat__heat__2005_01_01_00_00)
+2.1 variables(flow_in)(b__test_link_a_b_heat__heat__2005_01_01_01_00)
+0.023972602739726026 variables(flow_cap)(b__test_link_a_b_heat__heat)
+0.04794520547945205 variables(flow_cap)(b__test_supply_elec__electricity)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_link_a_b_elec__electricity) <= 10.0
   0 <= variables(flow_in)(a__test_link_a_b_heat__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_link_a_b_heat__heat__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_link_a_b_heat__heat) <= 5.0
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0 <= variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_link_a_b_elec__electricity__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_cap)(b__test_link_a_b_elec__electricity) <= 10.0
   0 <= variables(flow_in)(b__test_link_a_b_heat__heat__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_link_a_b_heat__heat__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_cap)(b__test_link_a_b_heat__heat) <= 5.0
   0.0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 10.0
end
