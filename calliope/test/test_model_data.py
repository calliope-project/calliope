import pytest

from calliope.preprocess.model_data import *
from calliope.core.attrdict import AttrDict


def model_run(coords=("('nodes', 'techs')")):
    _dict = {}

    if "('nodes', 'techs')" in coords:
        _dict.update({
            'locations': {
                'A': {
                    'techs': {
                        'foo': {
                            'constraints': {
                                'energy_cap_max': 1,
                                'energy_eff': 0.9
                            }
                        }
                    },
                    'links': {
                        'B': {
                            'techs': {
                                'bar': {
                                    'constraints': {
                                        'energy_cap_max': 1,
                                        'energy_eff': 0.9
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    if "('nodes',)" in coords:
        _dict["locations"]["A"]["available_area"] = 1
    if "('nodes', 'coordinates')" in coords:
        _dict["locations"]["A"]["coordinates"] = {"x": 1, "y": 1}
    if "('techs',)" in coords:
        _dict.update({
            "techs": {
                "foo": {
                    "inheritance": ["foobar", "supply"],
                    "essentials": {
                        "carrier": "electricity"
                    }
                },
                "bar": {
                    "inheritance": ["foobaz", "transmission"],
                    "essentials": {
                        "carrier": "electricity"
                    }
                }
            }
        })

    return AttrDict(_dict)

@pytest.mark.parametrize(
    "arg",
    ((), (["('nodes',)"]), (["('nodes', 'coordinates')"]), (["('nodes',)", "('nodes', 'coordinates')"]))
)
def test_get_node_param_keys(arg):
    param_dict = AttrDict()
    get_node_params(param_dict, model_run(("('nodes', 'techs')", *arg)))
    assert set(param_dict.keys()) == set(["('nodes', 'techs')", *arg])


def test_get_node_param_values():
    param_dict = AttrDict()
    get_node_params(param_dict, model_run(("('nodes', 'techs')", "('nodes',)", "('nodes', 'coordinates')")))

    assert set(param_dict["('nodes', 'techs')"].keys()) == set(["('A', 'foo')", "('B', 'bar:A')", "('A', 'bar:B')"])
    assert set(param_dict["('nodes',)"].keys()) == set(["('A',)"])
    assert set(param_dict["('nodes', 'coordinates')"].keys()) == set(["('A', 'x')", "('A', 'y')"])


def test_get_tech_params():
    param_dict = AttrDict()
    get_tech_params(param_dict, model_run(("('nodes', 'techs')", "('techs',)")))
    assert set(param_dict.keys()) == set(["('techs',)"])
    assert set(param_dict["('techs',)"].keys()) == set(["('foo',)", "('bar:B',)"])


def test_set_idx():
    key_str = set_idx("energy_cap", {'nodes': 'A', 'techs': 'foo'})
    assert key_str == "('nodes', 'techs').('A', 'foo').energy_cap"

def test_set_tech_at_node_info():
    pass

def test_set_tech_info():
    param_dict = AttrDict()
    set_tech_info(param_dict, ["foo"], "resource", 1)
    assert set(param_dict.keys()) == set(["('techs',)"])
    assert param_dict["('techs',)"]["('foo',)"]["resource"] == 1
