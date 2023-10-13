\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(system_balance)(0)_:
+1 variables(flow_out)(a__test_supply_coal__coal__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion_plus__coal__2005_01_01_00_00)
-1 variables(flow_in)(a__test_demand_coal__coal__2005_01_01_00_00)
+1 variables(fuel_distributor)(a__coal__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(1)_:
+1 variables(flow_out)(a__test_supply_coal__coal__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion_plus__coal__2005_01_01_01_00)
-1 variables(flow_in)(a__test_demand_coal__coal__2005_01_01_01_00)
+1 variables(fuel_distributor)(a__coal__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(2)_:
+1 variables(flow_out)(b__test_supply_coal__coal__2005_01_01_00_00)
-1 variables(flow_in)(b__test_conversion_plus__coal__2005_01_01_00_00)
-1 variables(flow_in)(b__test_demand_coal__coal__2005_01_01_00_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(3)_:
+1 variables(flow_out)(b__test_supply_coal__coal__2005_01_01_01_00)
-1 variables(flow_in)(b__test_conversion_plus__coal__2005_01_01_01_00)
-1 variables(flow_in)(b__test_demand_coal__coal__2005_01_01_01_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(4)_:
+1 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00)
-1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(5)_:
+1 variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00)
-1 variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(6)_:
+1 variables(flow_out)(a__test_supply_gas__gas__2005_01_01_00_00)
-1 variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(7)_:
+1 variables(flow_out)(a__test_supply_gas__gas__2005_01_01_01_00)
-1 variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(8)_:
+1 variables(flow_out)(b__test_conversion_plus__electricity__2005_01_01_00_00)
-1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(9)_:
+1 variables(flow_out)(b__test_conversion_plus__electricity__2005_01_01_01_00)
-1 variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00)
= 0.0

c_e_constraints(system_balance)(10)_:
+1 variables(flow_out)(b__test_supply_gas__gas__2005_01_01_00_00)
-1 variables(flow_in)(b__test_conversion_plus__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(system_balance)(11)_:
+1 variables(flow_out)(b__test_supply_gas__gas__2005_01_01_01_00)
-1 variables(flow_in)(b__test_conversion_plus__gas__2005_01_01_01_00)
= 0.0

c_e_constraints(restrict_total_imports_and_exports)(0)_:
+1 variables(fuel_distributor)(a__coal__2005_01_01_00_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_00_00)
= 0.0

c_e_constraints(restrict_total_imports_and_exports)(1)_:
+1 variables(fuel_distributor)(a__coal__2005_01_01_01_00)
+1 variables(fuel_distributor)(b__coal__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_out)(a__test_supply_coal__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion_plus__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_coal__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_coal__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion_plus__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_coal__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_coal__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion_plus__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_coal__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_coal__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion_plus__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_coal__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_conversion_plus__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_gas__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(a__test_supply_gas__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(a__test_conversion_plus__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_conversion_plus__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_conversion_plus__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_demand_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_gas__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion_plus__gas__2005_01_01_00_00) <= +inf
   0 <= variables(flow_out)(b__test_supply_gas__gas__2005_01_01_01_00) <= +inf
   0 <= variables(flow_in)(b__test_conversion_plus__gas__2005_01_01_01_00) <= +inf
end
