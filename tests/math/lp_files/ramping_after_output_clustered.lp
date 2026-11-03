\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_01_00)_:
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_02_00)_:
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_03_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_04_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_04_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_05_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_04_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_05_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_06_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_05_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_07_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_07_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_08_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_07_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_08_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_09_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_08_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_09_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_10_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_09_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_10_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_11_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_10_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_11_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_12_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_11_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_13_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_13_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_14_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_13_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_14_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_15_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_14_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_15_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_16_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_15_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_16_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_17_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_16_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_17_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_18_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_17_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_19_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_19_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_20_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_19_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_20_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_21_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_20_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_21_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_22_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_21_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_22_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_23_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_22_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_23_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_01_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_01_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_02_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_01_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_02_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_03_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_02_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_03_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_04_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_03_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_04_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_05_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_04_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_05_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_06_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_05_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_07_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_07_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_08_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_07_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_08_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_09_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_08_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_09_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_10_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_09_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_10_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_11_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_10_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_11_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_12_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_11_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_13_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_13_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_14_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_13_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_14_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_15_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_14_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_15_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_16_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_15_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_16_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_17_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_16_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_17_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_18_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_17_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_19_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_19_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_20_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_19_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_20_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_21_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_20_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_21_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_22_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_21_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_22_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_23_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_22_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_23_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_23_00)
<= 0.0

c_u_constraints(ramping_up_after_low_output)(a__test_supply_elec__electricity__2005_01_03_00_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_23_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_01_00)_:
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_02_00)_:
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_03_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_04_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_04_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_05_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_04_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_05_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_06_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_05_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_07_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_07_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_08_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_07_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_08_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_09_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_08_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_09_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_10_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_09_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_10_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_11_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_10_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_11_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_12_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_11_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_13_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_13_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_14_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_13_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_14_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_15_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_14_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_15_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_16_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_15_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_16_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_17_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_16_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_17_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_18_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_17_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_19_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_19_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_20_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_19_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_20_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_21_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_20_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_21_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_22_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_21_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_22_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_23_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_22_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_23_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_01_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_01_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_02_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_01_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_02_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_03_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_02_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_03_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_04_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_03_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_04_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_05_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_04_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_05_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_06_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_05_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_07_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_07_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_08_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_07_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_08_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_09_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_08_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_09_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_10_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_09_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_10_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_11_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_10_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_11_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_12_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_11_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_13_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_13_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_14_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_13_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_14_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_15_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_14_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_15_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_16_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_15_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_16_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_17_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_16_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_17_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_18_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_17_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_19_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_19_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_20_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_19_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_20_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_21_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_20_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_21_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_22_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_21_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_22_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_23_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_22_00)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_23_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_23_00)
<= 0.0

c_u_constraints(ramping_down_after_high_output)(a__test_supply_elec__electricity__2005_01_03_00_00)_:
-0.5 variables(flow_cap)(a__test_supply_elec__electricity)
-1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00)
+1.0 variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_23_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_02_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_03_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_04_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_05_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_06_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_07_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_08_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_09_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_10_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_11_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_12_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_13_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_14_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_15_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_16_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_17_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_18_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_19_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_20_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_21_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_22_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_23_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_01_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_00_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_02_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_03_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_04_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_05_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_06_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_07_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_08_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_09_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_10_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_11_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_12_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_13_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_14_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_15_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_16_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_17_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_18_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_19_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_20_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_21_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_22_00) <= +inf
   0.0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_03_23_00) <= +inf
end
