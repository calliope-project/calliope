\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+15.0 variables(flow_peak_month)(a__test_supply_elec__electricity__1)
+15.0 variables(flow_peak_month)(a__test_supply_elec__electricity__2)

s.t.

c_u_constraints(set_peak_month_flow)(a__test_supply_elec__electricity__2005_01_01_00_00__1)_:
-1 variables(flow_peak_month)(a__test_supply_elec__electricity__1)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
<= 0.0

c_u_constraints(set_peak_month_flow)(a__test_supply_elec__electricity__2005_01_01_01_00__1)_:
-1 variables(flow_peak_month)(a__test_supply_elec__electricity__1)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(set_peak_month_flow)(a__test_supply_elec__electricity__2005_01_01_02_00__2)_:
-1 variables(flow_peak_month)(a__test_supply_elec__electricity__2)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00)
<= 0.0

c_u_constraints(set_peak_month_flow)(a__test_supply_elec__electricity__2005_01_01_03_00__2)_:
-1 variables(flow_peak_month)(a__test_supply_elec__electricity__2)
+1 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00)
<= 0.0

bounds
   0 <= variables(flow_peak_month)(a__test_supply_elec__electricity__1) <= 10.0
   0 <= variables(flow_peak_month)(a__test_supply_elec__electricity__2) <= 10.0
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00) <= +inf
end

