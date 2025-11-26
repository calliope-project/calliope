\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+1.0 variables(purchased_units)(a__test_supply_elec)
+1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)

s.t.

c_u_constraints(piecewise_costs)(a__test_supply_elec__monetary__0)_:
-1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)
+5.0 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(piecewise_costs)(a__test_supply_elec__monetary__1)_:
-2.0 variables(purchased_units)(a__test_supply_elec)
-1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)
+7.0 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

c_u_constraints(piecewise_costs)(a__test_supply_elec__monetary__2)_:
-16.0 variables(purchased_units)(a__test_supply_elec)
-1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)
+14.0 variables(flow_cap)(a__test_supply_elec__electricity)
<= 0.0

bounds
   0.0 <= variables(purchased_units)(a__test_supply_elec) <= 1.0
   0 <= variables(piecewise_cost_investment)(a__test_supply_elec__monetary) <= +inf
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
binary
  variables(purchased_units)(a__test_supply_elec)
end
