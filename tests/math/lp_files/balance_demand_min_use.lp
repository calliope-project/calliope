\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_l_constraints(balance_demand_min_use)(a__test_demand_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00)
>= 5.0

c_l_constraints(balance_demand_min_use)(a__test_demand_elec__electricity__2005_01_01_01_00)_:
+1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00)
>= 5.0

c_l_constraints(balance_demand_min_use)(b__test_demand_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00)
>= 5.0

c_l_constraints(balance_demand_min_use)(b__test_demand_elec__electricity__2005_01_01_01_00)_:
+1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00)
>= 5.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
end
