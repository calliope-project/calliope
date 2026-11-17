\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_06_00)_:
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_12_00)_:
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_18_00)_:
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_06_00)_:
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_12_00)_:
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_18_00)_:
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_00_00)_:
-1.75 variables(flow_cap)(a__test_supply_elec__electricity)
+0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_06_00)_:
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_12_00)_:
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_18_00)_:
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_06_00)_:
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_12_00)_:
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_18_00)_:
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_00_00)_:
-6.75 variables(flow_cap)(a__test_supply_elec__electricity)
-0.16666666666666666 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00) <= +inf
end
