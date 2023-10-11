\* Source Pyomo model name=None *\

min
objectives(dummy_obj)(0):
+2.0 ONE_VAR_CONSTANT

s.t.

c_u_constraints(carrier_production_max)(a__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(carrier_prod)(a__test_supply_elec__electricity__2005_01_01_00_00)
-1.0 variables(energy_cap)(a__test_supply_elec)
<= 0.0

c_u_constraints(carrier_production_max)(a__test_supply_elec__electricity__2005_01_01_01_00)_:
-1.0 variables(energy_cap)(a__test_supply_elec)
+1 variables(carrier_prod)(a__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(carrier_production_max)(a__test_transmission_elec_b__electricity__2005_01_01_00_00)_:
+1 variables(carrier_prod)(a__test_transmission_elec_b__electricity__2005_01_01_00_00)
-1.0 variables(energy_cap)(a__test_transmission_elec_b)
<= 0.0

c_u_constraints(carrier_production_max)(a__test_transmission_elec_b__electricity__2005_01_01_01_00)_:
-1.0 variables(energy_cap)(a__test_transmission_elec_b)
+1 variables(carrier_prod)(a__test_transmission_elec_b__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(carrier_production_max)(a__test_transmission_heat_b__heat__2005_01_01_00_00)_:
+1 variables(carrier_prod)(a__test_transmission_heat_b__heat__2005_01_01_00_00)
-1.0 variables(energy_cap)(a__test_transmission_heat_b)
<= 0.0

c_u_constraints(carrier_production_max)(a__test_transmission_heat_b__heat__2005_01_01_01_00)_:
-1.0 variables(energy_cap)(a__test_transmission_heat_b)
+1 variables(carrier_prod)(a__test_transmission_heat_b__heat__2005_01_01_01_00)
<= 0.0

c_u_constraints(carrier_production_max)(b__test_supply_elec__electricity__2005_01_01_00_00)_:
+1 variables(carrier_prod)(b__test_supply_elec__electricity__2005_01_01_00_00)
-1.0 variables(energy_cap)(b__test_supply_elec)
<= 0.0

c_u_constraints(carrier_production_max)(b__test_supply_elec__electricity__2005_01_01_01_00)_:
-1.0 variables(energy_cap)(b__test_supply_elec)
+1 variables(carrier_prod)(b__test_supply_elec__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(carrier_production_max)(b__test_transmission_elec_a__electricity__2005_01_01_00_00)_:
+1 variables(carrier_prod)(b__test_transmission_elec_a__electricity__2005_01_01_00_00)
-1.0 variables(energy_cap)(b__test_transmission_elec_a)
<= 0.0

c_u_constraints(carrier_production_max)(b__test_transmission_elec_a__electricity__2005_01_01_01_00)_:
-1.0 variables(energy_cap)(b__test_transmission_elec_a)
+1 variables(carrier_prod)(b__test_transmission_elec_a__electricity__2005_01_01_01_00)
<= 0.0

c_u_constraints(carrier_production_max)(b__test_transmission_heat_a__heat__2005_01_01_00_00)_:
+1 variables(carrier_prod)(b__test_transmission_heat_a__heat__2005_01_01_00_00)
-1.0 variables(energy_cap)(b__test_transmission_heat_a)
<= 0.0

c_u_constraints(carrier_production_max)(b__test_transmission_heat_a__heat__2005_01_01_01_00)_:
-1.0 variables(energy_cap)(b__test_transmission_heat_a)
+1 variables(carrier_prod)(b__test_transmission_heat_a__heat__2005_01_01_01_00)
<= 0.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0 <= variables(carrier_prod)(a__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   100.0 <= variables(energy_cap)(a__test_supply_elec) <= 100.0
   0 <= variables(carrier_prod)(a__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(carrier_prod)(a__test_transmission_elec_b__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(energy_cap)(a__test_transmission_elec_b) <= 10.0
   0 <= variables(carrier_prod)(a__test_transmission_elec_b__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(carrier_prod)(a__test_transmission_heat_b__heat__2005_01_01_00_00) <= +inf
   0.0 <= variables(energy_cap)(a__test_transmission_heat_b) <= 5.0
   0 <= variables(carrier_prod)(a__test_transmission_heat_b__heat__2005_01_01_01_00) <= +inf
   0 <= variables(carrier_prod)(b__test_supply_elec__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(energy_cap)(b__test_supply_elec) <= 10.0
   0 <= variables(carrier_prod)(b__test_supply_elec__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(carrier_prod)(b__test_transmission_elec_a__electricity__2005_01_01_00_00) <= +inf
   0.0 <= variables(energy_cap)(b__test_transmission_elec_a) <= 10.0
   0 <= variables(carrier_prod)(b__test_transmission_elec_a__electricity__2005_01_01_01_00) <= +inf
   0 <= variables(carrier_prod)(b__test_transmission_heat_a__heat__2005_01_01_00_00) <= +inf
   0.0 <= variables(energy_cap)(b__test_transmission_heat_a) <= 5.0
   0 <= variables(carrier_prod)(b__test_transmission_heat_a__heat__2005_01_01_01_00) <= +inf
end
