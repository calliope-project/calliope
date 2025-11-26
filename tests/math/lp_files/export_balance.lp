\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(export_balance)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_00_00)
-1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(export_balance)(a__test_supply_elec__electricity__2005_01_01_01_00)_:
+1 variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_01_00)
-1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(export_balance)(b__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_export)(b__test_supply_elec__electricity__2005_01_01_00_00)
-1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(export_balance)(b__test_supply_elec__electricity__2005_01_01_01_00)_:
+1 variables(flow_export)(b__test_supply_elec__electricity__2005_01_01_01_00)
-1 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_00_00) <= 5.0
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_export)(a__test_supply_elec__electricity__2005_01_01_01_00) <= 5.0
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_export)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_export)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
end
