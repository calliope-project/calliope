\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(unit_capacity_max_systemwide_milp)(test_supply_elec)_:
+1 variables(purchased_units)(a__test_supply_elec)
<= 5.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(purchased_units)(a__test_supply_elec) <= 1.0
binary
  variables(purchased_units)(a__test_supply_elec)
end
