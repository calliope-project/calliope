\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(annual_capacity_factor_min)(a__test_supply_elec__electricity)_:
+1.6 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(annual_capacity_factor_min)(b__test_supply_elec__electricity)_:
+1.6 variables(flow_cap)(b__test_supply_elec__electricity)
-1.0 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
-1.0 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(annual_capacity_factor_max)(a__test_supply_elec__electricity)_:
-1.8 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(annual_capacity_factor_max)(b__test_supply_elec__electricity)_:
-1.8 variables(flow_cap)(b__test_supply_elec__electricity)
+1.0 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
+1.0 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 10.0
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
end
