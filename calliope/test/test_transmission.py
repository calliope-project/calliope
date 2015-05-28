from io import StringIO
import pytest

from calliope import transmission, utils


class TestTransmission:
    @pytest.fixture
    def transmission_network(self):
        setup = StringIO("""
        links:
            1,2:
                hvac:
                    constraints:
                        e_cap.max: 10
            3,2:
                hvac:
                    constraints:
                        e_cap.max: 20
        """)
        return utils.AttrDict.from_yaml(setup)

    @pytest.fixture
    def transmission_network_erroneous(self):
        setup = StringIO("""
        links:
            1,2:
                hvac:
                    constraints:
                        e_cap.max: 10
            3,2:
                hvac:
                    constraints:
                        e_cap.max: 20
            4,4:
        """)
        return utils.AttrDict.from_yaml(setup)

    def test_get_remotes(self):
        y = 'hvac:b'
        x = 'a'
        assert transmission.get_remotes(y, x) == ('hvac:a', 'b')

    def test_get_transmission_techs(self, transmission_network):
        links = transmission_network.links
        techs = transmission.get_transmission_techs(links)
        assert sorted(techs) == ['hvac:1', 'hvac:2', 'hvac:3']

    def test_get_transmission_techs_empty(self):
        links = {}
        techs = transmission.get_transmission_techs(links)
        assert techs == []

    def test_explode_transmission_tree(self, transmission_network):
        links = transmission_network.links
        possible_x = ['1', '2', '3']
        tree = transmission.explode_transmission_tree(links, possible_x)
        assert tree == {'1': {'hvac:2': {'constraints': {'e_cap': {'max': 10}}}},
                        '3': {'hvac:2': {'constraints': {'e_cap': {'max': 20}}}},
                        '2': {'hvac:3': {'constraints': {'e_cap': {'max': 20}}},
                        'hvac:1': {'constraints': {'e_cap': {'max': 10}}}}}

    def test_explode_transmission_tree_invalid_x(self, transmission_network):
        links = transmission_network.links
        possible_x = ['1', '2']
        with pytest.raises(KeyError):
            transmission.explode_transmission_tree(links, possible_x)

    def test_explode_transmission_tree_bad_links(self,
                                                 transmission_network_erroneous):
        links = transmission_network_erroneous.links
        possible_x = ['1', '2', '3', '4']
        with pytest.raises(KeyError):
            transmission.explode_transmission_tree(links, possible_x)

    def explode_transmission_tree_empty(self):
        links = {}
        possible_x = ['1', '2']
        tree = transmission.explode_transmission_tree(links, possible_x)
        assert tree is None
