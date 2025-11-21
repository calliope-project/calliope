\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+1000.0 variables(purchased_units)(a__test_supply_elec)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   1.0 <= variables(purchased_units)(a__test_supply_elec) <= 2.0
general
  variables(purchased_units)(a__test_supply_elec)
end
