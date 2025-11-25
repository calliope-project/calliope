\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+0.5 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00)
+0.5 variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00)
+0.011506849315068492 variables(flow_cap)(a__test_supply_elec__electricity)
+0.5 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00)
+0.5 variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00)
+0.011506849315068492 variables(flow_cap)(b__test_supply_elec__electricity)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0.0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 10.0
end
