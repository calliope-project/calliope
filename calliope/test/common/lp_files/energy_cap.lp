\* Source Pyomo model name=None *\

min
objectives(foo)(0):
+1 variables(energy_cap)(a__test_supply_elec)
+1 variables(energy_cap)(b__test_supply_elec)

s.t.

c_e_ONE_VAR_CONSTANT:
ONE_VAR_CONSTANT = 1.0

bounds
   1 <= variables(energy_cap)(a__test_supply_elec) <= +inf
   0 <= variables(energy_cap)(b__test_supply_elec) <= 100
end
