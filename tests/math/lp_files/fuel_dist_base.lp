\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(system_balance)(a__coal__2005_01_01_00_00)_:
+1 variables(flow_out)(a__test_supply_coal__coal__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__coal__2005_01_01_00_00)
-1 variables(flow_in)(a__test_demand_coal__coal__2005_01_01_00_00)
+1 variables(fuel_distributor)(a__coal__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(a__coal__2005_01_01_01_00)_:
+1 variables(flow_out)(a__test_supply_coal__coal__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__coal__2005_01_01_01_00)
-1 variables(flow_in)(a__test_demand_coal__coal__2005_01_01_01_00)
+1 variables(fuel_distributor)(a__coal__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(b__coal__2005_01_01_00_00)_:
+1 variables(flow_out)(b__test_supply_coal__coal__2005_01_01_00_00)
-1 variables(flow_in)(b__test_conversion__coal__2005_01_01_00_00)
-1 variables(flow_in)(b__test_demand_coal__coal__2005_01_01_00_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(b__coal__2005_01_01_01_00)_:
+1 variables(flow_out)(b__test_supply_coal__coal__2005_01_01_01_00)
-1 variables(flow_in)(b__test_conversion__coal__2005_01_01_01_00)
-1 variables(flow_in)(b__test_demand_coal__coal__2005_01_01_01_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(a__electricity__2005_01_01_00_00)_:
+1 variables(flow_out)(a__test_conversion__electricity__2005_01_01_00_00)
-1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(a__electricity__2005_01_01_01_00)_:
+1 variables(flow_out)(a__test_conversion__electricity__2005_01_01_01_00)
-1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(a__gas__2005_01_01_00_00)_:
+1 variables(flow_out)(a__test_supply_gas__gas__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(a__gas__2005_01_01_01_00)_:
+1 variables(flow_out)(a__test_supply_gas__gas__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(b__electricity__2005_01_01_00_00)_:
+1 variables(flow_out)(b__test_conversion__electricity__2005_01_01_00_00)
-1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(b__electricity__2005_01_01_01_00)_:
+1 variables(flow_out)(b__test_conversion__electricity__2005_01_01_01_00)
-1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(b__gas__2005_01_01_00_00)_:
+1 variables(flow_out)(b__test_supply_gas__gas__2005_01_01_00_00)
-1 variables(flow_in)(b__test_conversion__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(b__gas__2005_01_01_01_00)_:
+1 variables(flow_out)(b__test_supply_gas__gas__2005_01_01_01_00)
-1 variables(flow_in)(b__test_conversion__gas__2005_01_01_01_00)
= 0.0

c_e_constraints(restrict_total_imports_and_exports)(coal__2005_01_01_00_00)_:
+1 variables(fuel_distributor)(a__coal__2005_01_01_00_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_00_00)
= 0.0

c_e_constraints(restrict_total_imports_and_exports)(coal__2005_01_01_01_00)_:
+1 variables(fuel_distributor)(a__coal__2005_01_01_01_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_coal__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_coal__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_coal__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_coal__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_coal__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_coal__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_coal__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_coal__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_gas__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_gas__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_conversion__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_conversion__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_gas__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_gas__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion__gas__2005_01_01_01_00) <= +inf
end
