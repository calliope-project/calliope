\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(piecewise_costs)(a__test_supply_elec__monetary__0)_:
+5.0 variables(flow_cap)(a__test_supply_elec__electricity)
-1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)
<= 0.0

c_u_constraints(piecewise_costs)(a__test_supply_elec__monetary__1)_:
+7.0 variables(flow_cap)(a__test_supply_elec__electricity)
-1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)
-2.0 variables(purchased)(a__test_supply_elec)
<= 0.0

c_u_constraints(piecewise_costs)(a__test_supply_elec__monetary__2)_:
+14.0 variables(flow_cap)(a__test_supply_elec__electricity)
-1 variables(piecewise_cost_investment)(a__test_supply_elec__monetary)
-16.0 variables(purchased)(a__test_supply_elec)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0 <= variables(piecewise_cost_investment)(a__test_supply_elec__monetary) <= +inf
   0 <= variables(purchased)(a__test_supply_elec) <= 1
binary
  variables(purchased)(a__test_supply_elec)
end
