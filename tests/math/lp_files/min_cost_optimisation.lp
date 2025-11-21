\* Source Pyomo model name=None *\

min
objectives(min_cost_optimisation)(0):
+0.00012576263948632513 variables(flow_cap)(a__test_link_a_b_elec__electricity)
+0.00012576263948632513 variables(flow_cap)(a__test_link_a_b_heat__heat)
+0.00025152527897265027 variables(flow_cap)(a__test_supply_elec__electricity)
+0.00012576263948632513 variables(flow_cap)(b__test_link_a_b_elec__electricity)
+0.00012576263948632513 variables(flow_cap)(b__test_link_a_b_heat__heat)
+0.00025152527897265027 variables(flow_cap)(b__test_supply_elec__electricity)
+1000.0 variables(unmet_demand)(a__electricity__2005_01_01_00_00)
-1000.0 variables(unused_supply)(a__electricity__2005_01_01_00_00)
+1000.0 variables(unmet_demand)(a__heat__2005_01_01_00_00)
-1000.0 variables(unused_supply)(a__heat__2005_01_01_00_00)
+1000.0 variables(unmet_demand)(b__electricity__2005_01_01_00_00)
-1000.0 variables(unused_supply)(b__electricity__2005_01_01_00_00)
+1000.0 variables(unmet_demand)(b__heat__2005_01_01_00_00)
-1000.0 variables(unused_supply)(b__heat__2005_01_01_00_00)
+1000.0 variables(unmet_demand)(a__electricity__2005_01_01_01_00)
-1000.0 variables(unused_supply)(a__electricity__2005_01_01_01_00)
+1000.0 variables(unmet_demand)(a__heat__2005_01_01_01_00)
-1000.0 variables(unused_supply)(a__heat__2005_01_01_01_00)
+1000.0 variables(unmet_demand)(b__electricity__2005_01_01_01_00)
-1000.0 variables(unused_supply)(b__electricity__2005_01_01_01_00)
+1000.0 variables(unmet_demand)(b__heat__2005_01_01_01_00)
-1000.0 variables(unused_supply)(b__heat__2005_01_01_01_00)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_cap)(a__test_link_a_b_elec__electricity) <= 10.0
   0.0 <= variables(flow_cap)(a__test_link_a_b_heat__heat) <= 5.0
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0.0 <= variables(flow_cap)(b__test_link_a_b_elec__electricity) <= 10.0
   0.0 <= variables(flow_cap)(b__test_link_a_b_heat__heat) <= 5.0
   0.0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 10.0
   0 <= variables(unmet_demand)(a__electricity__2005_01_01_00_00) <= +inf
   -inf <= variables(unused_supply)(a__electricity__2005_01_01_00_00) <= 0
   0 <= variables(unmet_demand)(a__heat__2005_01_01_00_00) <= +inf
   -inf <= variables(unused_supply)(a__heat__2005_01_01_00_00) <= 0
   0 <= variables(unmet_demand)(b__electricity__2005_01_01_00_00) <= +inf
   -inf <= variables(unused_supply)(b__electricity__2005_01_01_00_00) <= 0
   0 <= variables(unmet_demand)(b__heat__2005_01_01_00_00) <= +inf
   -inf <= variables(unused_supply)(b__heat__2005_01_01_00_00) <= 0
   0 <= variables(unmet_demand)(a__electricity__2005_01_01_01_00) <= +inf
   -inf <= variables(unused_supply)(a__electricity__2005_01_01_01_00) <= 0
   0 <= variables(unmet_demand)(a__heat__2005_01_01_01_00) <= +inf
   -inf <= variables(unused_supply)(a__heat__2005_01_01_01_00) <= 0
   0 <= variables(unmet_demand)(b__electricity__2005_01_01_01_00) <= +inf
   -inf <= variables(unused_supply)(b__electricity__2005_01_01_01_00) <= 0
   0 <= variables(unmet_demand)(b__heat__2005_01_01_01_00) <= +inf
   -inf <= variables(unused_supply)(b__heat__2005_01_01_01_00) <= 0
end
