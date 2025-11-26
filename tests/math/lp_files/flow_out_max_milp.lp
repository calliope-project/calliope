\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(flow_out_max_milp)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-10.0 variables(operating_units)(a__test_supply_elec__2005_01_01_00_00)
<= 0.0

c_u_constraints(flow_out_max_milp)(a__test_supply_elec__electricity__2005_01_01_01_00)_:
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
-10.0 variables(operating_units)(a__test_supply_elec__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(operating_units)(a__test_supply_elec__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(operating_units)(a__test_supply_elec__2005_01_01_01_00) <= +inf
general
  variables(operating_units)(a__test_supply_elec__2005_01_01_00_00)
  variables(operating_units)(a__test_supply_elec__2005_01_01_01_00)
end
