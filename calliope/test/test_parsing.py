from io import StringIO

import pytest
import ruamel.yaml as yaml

from calliope.backend import parsing, equation_parser


class TestEquationParser:
    def test_parse_function_single_arg(self):
        test_string = "resource_eff[node, tech, timestep] * get_index(timestep)"
        parser = equation_parser.setup_parser()
        result = parser.parse_string(test_string)
        assert result[0].value[-1].args[0].dump() == "['timestep']"

    def test_parse_function_two_args(self):
        test_string = "resource_eff[node, tech, timestep] * get_index(x=2, timestep)"
        parser = equation_parser.setup_parser()
        result = parser.parse_string(test_string)
        assert result[0].value[-1].args[0].dump() == "['x', CONST:2]"


class TestParsing:
    @pytest.fixture
    def full_constraint_test_data(self):
        setup_string = """
        constraints:
            balance_supply:
                foreach: [carrier in carriers, node in nodes, tech in techs, timestep in timesteps]
                where: [resource, and, inheritance(supply)]
                equations:
                    - where: [energy_eff=0]
                      expression: carrier_prod[carrier, node, tech, timestep] == 0
                    - where: [force_resource=True, and, not energy_eff=0]
                      expression: $carrier_prod_div_energy_eff == $available_resource
                    - where: [resource_min_use, and, not force_resource=True, and, not energy_eff=0]
                      expression: resource_min_use[node, tech] * $available_resource <= $carrier_prod_div_energy_eff <= $available_resource
                    - where: [not resource_min_use, and, not force_resource=True, and, not energy_eff=0]
                      expression: $carrier_prod_div_energy_eff <= $available_resource
                components:
                    available_resource:
                        - where: [resource_unit='energy_per_area']
                          expression: resource[node, tech, timestep] * resource_scale[node, tech] * resource_area[node, tech]
                        - where: [resource_unit='energy_per_cap']
                          expression: resource[node, tech, timestep] * resource_scale[node, tech] * energy_cap[node, tech]
                        - where: [resource_unit='energy']
                          expression: resource[node, tech, timestep] * resource_scale[node, tech]
                    carrier_prod_div_energy_eff:
                        - expression: carrier_prod[carrier, node, tech, timestep] / energy_eff[node, tech, timestep]
        """
        yaml_loader = yaml.YAML(typ="safe", pure=True)
        return yaml_loader.load(StringIO(setup_string))

    def test_find_and_replace_components(self):
        pass

    def test_get_variant_equations(self):
        pass

    def test_get_variant_equations_complex_expression(self):
        #
        pass

    def test_get_variant_equations_zero_expression(self):
        # expression: 0
        pass

    def test_parse_foreach(self):
        pass

    def test_parse_constraint_equations(self):
        pass

    def test_full_parsing(self, full_constraint_test_data):
        constraints = full_constraint_test_data["constraints"]
        constraint_name = "balance_supply"
        result = parsing.process_constraint(constraints, constraint_name)

        # FIXME this should do more than just check a couple name strings
        # FIXME or there need to be more unit tests for specific functionality...
        assert result["name"] == "balance_supply"
        assert result["equations"][1]["name"] == "balance_supply_1_0"
