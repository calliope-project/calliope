\* Source Pyomo model name=None *\

min
objectives(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_e_constraints(balance_conversion)(0)_:
+1 variables(carrier_prod)(a__test_conversion__heat__2005_01_01_00_00)
+0.9 variables(carrier_con)(a__test_conversion__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_conversion)(1)_:
+1 variables(carrier_prod)(a__test_conversion__heat__2005_01_01_01_00)
+0.9 variables(carrier_con)(a__test_conversion__gas__2005_01_01_01_00)
= 0.0

c_e_constraints(balance_conversion)(2)_:
+1 variables(carrier_prod)(b__test_conversion__heat__2005_01_01_00_00)
+0.9 variables(carrier_con)(b__test_conversion__gas__2005_01_01_00_00)
= 0.0

c_e_constraints(balance_conversion)(3)_:
+1 variables(carrier_prod)(b__test_conversion__heat__2005_01_01_01_00)
+0.9 variables(carrier_con)(b__test_conversion__gas__2005_01_01_01_00)
= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(carrier_prod)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   -inf <= variables(carrier_con)(a__test_conversion__gas__2005_01_01_00_00) <= 0
   0 <= variables(carrier_prod)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
   -inf <= variables(carrier_con)(a__test_conversion__gas__2005_01_01_01_00) <= 0
   0 <= variables(carrier_prod)(b__test_conversion__heat__2005_01_01_00_00) <= +inf
   -inf <= variables(carrier_con)(b__test_conversion__gas__2005_01_01_00_00) <= 0
   0 <= variables(carrier_prod)(b__test_conversion__heat__2005_01_01_01_00) <= +inf
   -inf <= variables(carrier_con)(b__test_conversion__gas__2005_01_01_01_00) <= 0
end
