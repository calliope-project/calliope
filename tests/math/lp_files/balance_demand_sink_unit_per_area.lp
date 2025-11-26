\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(balance_demand)(a__test_demand_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00)
-5.0 variables(area_use)(a__test_demand_elec)
= 0.0

c_e_constraints(balance_demand)(a__test_demand_elec__electricity__2005_01_01_01_00)_:
-5.0 variables(area_use)(a__test_demand_elec)
+1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00)
= 0.0

c_e_constraints(balance_demand)(b__test_demand_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00)
-5.0 variables(area_use)(b__test_demand_elec)
= 0.0

c_e_constraints(balance_demand)(b__test_demand_elec__electricity__2005_01_01_01_00)_:
-5.0 variables(area_use)(b__test_demand_elec)
+1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(area_use)(a__test_demand_elec) <= 50.0
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(area_use)(b__test_demand_elec) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
end
