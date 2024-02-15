\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_c(0)_:
+1.0 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(1)
+2.0 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(2)
+10.0 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(3)
-1 variables(flow_cap)(a__test_supply_elec__electricity)
= 0

c_e_piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_c(1)_:
+5.0 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(1)
+6.0 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(2)
+20.0 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(3)
-1 variables(piecewise_cost_investment)(a__test_supply_elec__electricity__monetary)
= 0

c_e_piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_c(2)_:
+1 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(1)
+1 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(2)
+1 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(3)
+1 piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(0)
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(1) <= +inf
   0 <= piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(2) <= +inf
   0 <= piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(3) <= +inf
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= 10.0
   0 <= variables(piecewise_cost_investment)(a__test_supply_elec__electricity__monetary) <= +inf
   0 <= piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(0) <= +inf
SOS

piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_s: S2::
  piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(0):1
  piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(1):2
  piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(2):3
  piecewise_constraints(sos2_piecewise_costs)(a__test_supply_elec__electricity__monetary)_v(3):4

end
