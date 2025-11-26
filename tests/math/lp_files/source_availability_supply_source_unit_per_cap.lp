\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(source_availability_supply)(a__test_supply_plus__2005_01_01_00_00)_:
+1 variables(source_use)(a__test_supply_plus__2005_01_01_00_00)
-19.200000000000003 variables(flow_cap)(a__test_supply_plus__electricity)
<= 0.0

c_u_constraints(source_availability_supply)(a__test_supply_plus__2005_01_02_00_00)_:
-19.200000000000003 variables(flow_cap)(a__test_supply_plus__electricity)
+1 variables(source_use)(a__test_supply_plus__2005_01_02_00_00)
<= 0.0

c_u_constraints(source_availability_supply)(b__test_supply_plus__2005_01_01_00_00)_:
+1 variables(source_use)(b__test_supply_plus__2005_01_01_00_00)
-80.0 variables(flow_cap)(b__test_supply_plus__electricity)
<= 0.0

c_u_constraints(source_availability_supply)(b__test_supply_plus__2005_01_02_00_00)_:
-80.0 variables(flow_cap)(b__test_supply_plus__electricity)
+1 variables(source_use)(b__test_supply_plus__2005_01_02_00_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(source_use)(a__test_supply_plus__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(a__test_supply_plus__electricity) <= 15.0
   0 <= variables(source_use)(a__test_supply_plus__2005_01_02_00_00) <= +inf
   0 <= variables(source_use)(b__test_supply_plus__2005_01_01_00_00) <= +inf
   0.0 <= variables(flow_cap)(b__test_supply_plus__electricity) <= 15.0
   0 <= variables(source_use)(b__test_supply_plus__2005_01_02_00_00) <= +inf
end
