\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2 ONE_VAR_CONSTANT

s.t.

c_e_constraints(balance_conversion)(0)_:
+0.90000000000000002 variables(carrier_con)(a__test_conversion__gas__2005_01_01_00_00)
+1 variables(carrier_prod)(a__test_conversion__heat__2005_01_01_00_00)
= 0

c_e_constraints(balance_conversion)(1)_:
+0.90000000000000002 variables(carrier_con)(a__test_conversion__gas__2005_01_01_01_00)
+1 variables(carrier_prod)(a__test_conversion__heat__2005_01_01_01_00)
= 0

c_e_constraints(balance_conversion)(2)_:
+0.90000000000000002 variables(carrier_con)(b__test_conversion__gas__2005_01_01_00_00)
+1 variables(carrier_prod)(b__test_conversion__heat__2005_01_01_00_00)
= 0

c_e_constraints(balance_conversion)(3)_:
+0.90000000000000002 variables(carrier_con)(b__test_conversion__gas__2005_01_01_01_00)
+1 variables(carrier_prod)(b__test_conversion__heat__2005_01_01_01_00)
= 0

c_e_ONE_VAR_CONSTANT:
ONE_VAR_CONSTANT = 1.0

bounds
   0 <= variables(carrier_prod)(a__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(carrier_prod)(a__test_conversion__heat__2005_01_01_01_00) <= +inf
   0 <= variables(carrier_prod)(b__test_conversion__heat__2005_01_01_00_00) <= +inf
   0 <= variables(carrier_prod)(b__test_conversion__heat__2005_01_01_01_00) <= +inf
    -inf <= variables(carrier_con)(a__test_conversion__gas__2005_01_01_00_00) <= 0
    -inf <= variables(carrier_con)(a__test_conversion__gas__2005_01_01_01_00) <= 0
    -inf <= variables(carrier_con)(b__test_conversion__gas__2005_01_01_00_00) <= 0
    -inf <= variables(carrier_con)(b__test_conversion__gas__2005_01_01_01_00) <= 0
end
