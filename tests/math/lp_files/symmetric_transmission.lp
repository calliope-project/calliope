\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(symmetric_transmission)(test_link_a_b_elec__electricity)_:
+1 variables(flow_cap)(a__test_link_a_b_elec__electricity)
-1 variables(flow_cap)(b__test_link_a_b_elec__electricity)
= 0.0

c_e_constraints(symmetric_transmission)(test_link_a_b_heat__heat)_:
+1 variables(flow_cap)(a__test_link_a_b_heat__heat)
-1 variables(flow_cap)(b__test_link_a_b_heat__heat)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_cap)(a__test_link_a_b_elec__electricity) <= 10.0
   0.0 <= variables(flow_cap)(b__test_link_a_b_elec__electricity) <= 10.0
   0.0 <= variables(flow_cap)(a__test_link_a_b_heat__heat) <= 5.0
   0.0 <= variables(flow_cap)(b__test_link_a_b_heat__heat) <= 5.0
end
