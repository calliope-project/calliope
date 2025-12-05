\* Source Pyomo model name=None *\

min
objectives(test_obj)(0):
+10.0 variables(flow_cap)(a__test_supply_plus__electricity)
+50.0 variables(storage_cap)(a__test_supply_plus)
+30.0 variables(source_cap)(a__test_supply_plus)
+20.0 variables(area_use)(a__test_supply_plus)
+10.0 variables(flow_cap)(b__test_supply_plus__electricity)
+50.0 variables(storage_cap)(b__test_supply_plus)
+30.0 variables(source_cap)(b__test_supply_plus)
+20.0 variables(area_use)(b__test_supply_plus)

s.t.

c_e_ONE_VAR_CONSTANT:
+1 ONE_VAR_CONSTANT
= 1

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= variables(flow_cap)(a__test_supply_plus__electricity) <= 15.0
   0.0 <= variables(storage_cap)(a__test_supply_plus) <= 35.0
   0.0 <= variables(source_cap)(a__test_supply_plus) <= +inf
   0.0 <= variables(area_use)(a__test_supply_plus) <= +inf
   0.0 <= variables(flow_cap)(b__test_supply_plus__electricity) <= 15.0
   0.0 <= variables(storage_cap)(b__test_supply_plus) <= 35.0
   0.0 <= variables(source_cap)(b__test_supply_plus) <= +inf
   0.0 <= variables(area_use)(b__test_supply_plus) <= +inf
end
