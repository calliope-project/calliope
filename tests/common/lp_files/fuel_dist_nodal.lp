\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(restrict_nodal_imports)(0)_:
+1 variables(fuel_distributor)(b__coal__2005_01_01_00_00)
<= 5.0

c_u_constraints(restrict_nodal_imports)(1)_:
+1 variables(fuel_distributor)(b__coal__2005_01_01_01_00)
<= 5.0

c_u_constraints(restrict_nodal_exports)(0)_:
-1.0 variables(fuel_distributor)(a__coal__2005_01_01_00_00)
<= 3.0

c_u_constraints(restrict_nodal_exports)(1)_:
-1.0 variables(fuel_distributor)(a__coal__2005_01_01_01_00)
<= 3.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(b__coal__2005_01_01_01_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_00_00) <= +inf
   -inf <= variables(fuel_distributor)(a__coal__2005_01_01_01_00) <= +inf
end
