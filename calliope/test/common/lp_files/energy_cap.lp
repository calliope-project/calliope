\* Source Pyomo model name=None *\

min
objectives(0):
+1 variables(energy_cap)(test_supply_elec__a)
+1 variables(energy_cap)(test_supply_elec__b)

s.t.

c_e_ONE_VAR_CONSTANT:
ONE_VAR_CONSTANT = 1.0

bounds
   1 <= variables(energy_cap)(test_supply_elec__a) <= +inf
   0 <= variables(energy_cap)(test_supply_elec__b) <= 100
end
