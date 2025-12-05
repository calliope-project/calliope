\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+255.0 variables(purchased_units)(a__test_link_a_b_elec)
+255.0 variables(purchased_units)(b__test_link_a_b_elec)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(purchased_units)(a__test_link_a_b_elec) <= +inf
   0.0 <= variables(purchased_units)(b__test_link_a_b_elec) <= +inf
general
  variables(purchased_units)(a__test_link_a_b_elec)
  variables(purchased_units)(b__test_link_a_b_elec)
end
