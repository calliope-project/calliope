\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+1 variables(flow_cap)(a__test_supply_elec__electricity)
+1 variables(flow_cap)(b__test_supply_elec__electricity)

s.t.

c_l_constraints(flow_capacity_minimum)(a__test_supply_elec__electricity)_:
+1 variables(flow_cap)(a__test_supply_elec__electricity)
>= 1.0

bounds
   0 <= variables(flow_cap)(a__test_supply_elec__electricity) <= +inf
   0 <= variables(flow_cap)(b__test_supply_elec__electricity) <= 100.0
end
