from re import L
import pytest

from calliope.preprocess.model_data import (
    get_node_params,
    get_tech_params,
    set_idx,
    set_tech_at_node_info,
    set_tech_info,
    add_attributes,
)
from calliope.core.attrdict import AttrDict
from calliope._version import __version__


class TestModelData:
    @pytest.fixture
    def model_run(self):
        def _model_run(coords=("('nodes', 'techs')")):
            _dict = {}

            if "('nodes', 'techs')" in coords:
                _dict.update(
                    {
                        "locations": {
                            "A": {
                                "techs": {
                                    "foo": {
                                        "constraints": {
                                            "energy_cap_max": 1,
                                            "energy_eff": 0.9,
                                        },
                                        "switches": {"resource_unit": "bar"},
                                    }
                                },
                                "links": {
                                    "B": {
                                        "techs": {
                                            "bar": {
                                                "constraints": {
                                                    "energy_cap_max": 1,
                                                    "energy_eff": 0.9,
                                                }
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    }
                )
            if "('nodes',)" in coords:
                _dict["locations"]["A"]["available_area"] = 1
            if "('nodes', 'coordinates')" in coords:
                _dict["locations"]["A"]["coordinates"] = {"x": 1, "y": 1}
            if "('techs',)" in coords:
                _dict.update(
                    {
                        "techs": {
                            "foo": {
                                "inheritance": ["foobar", "supply"],
                                "essentials": {"carrier": "electricity"},
                            },
                            "bar": {
                                "inheritance": ["foobaz", "transmission"],
                                "essentials": {"carrier": "electricity"},
                            },
                        }
                    }
                )
            if "('nodes', 'techs', 'costs')" in coords:
                _dict.update({"locations.A.techs.foo.costs.monetary.energy_cap": 2})

            return AttrDict(_dict)

        return _model_run

    @pytest.mark.parametrize(
        "arg",
        (
            (),
            (["('nodes',)"]),
            (["('nodes', 'coordinates')"]),
            (["('nodes',)", "('nodes', 'coordinates')"]),
        ),
    )
    def test_get_node_param_keys(self, model_run, arg):
        param_dict = AttrDict()
        get_node_params(param_dict, model_run(("('nodes', 'techs')", *arg)))
        assert set(param_dict.keys()) == set(["('nodes', 'techs')", *arg])

    def test_get_node_param_values(self, model_run):
        param_dict = AttrDict()
        get_node_params(
            param_dict,
            model_run(("('nodes', 'techs')", "('nodes',)", "('nodes', 'coordinates')")),
        )

        assert set(param_dict["('nodes', 'techs')"].keys()) == set(
            ["('A', 'foo')", "('B', 'bar:A')", "('A', 'bar:B')"]
        )
        assert set(param_dict["('nodes',)"].keys()) == set(["('A',)"])
        assert set(param_dict["('nodes', 'coordinates')"].keys()) == set(
            ["('A', 'x')", "('A', 'y')"]
        )

    def test_get_tech_params(self, model_run):
        param_dict = AttrDict()
        get_tech_params(param_dict, model_run(("('nodes', 'techs')", "('techs',)")))
        assert set(param_dict.keys()) == set(["('techs',)"])
        assert set(param_dict["('techs',)"].keys()) == set(["('foo',)", "('bar:B',)"])

    def test_set_idx(self):
        key_str = set_idx("energy_cap", {"nodes": "A", "techs": "foo"})
        assert key_str == "('nodes', 'techs').('A', 'foo').energy_cap"

    def test_set_tech_at_node_info(self, model_run):
        param_dict = AttrDict()
        coords = ["('nodes', 'techs')", "('nodes', 'techs', 'costs')"]
        set_tech_at_node_info(
            param_dict, "A", "foo", model_run(coords).locations.A.techs.foo
        )
        assert set(param_dict.keys()) == set(coords)
        assert param_dict[coords[0]]["('A', 'foo')"]["energy_cap_max"] == 1
        assert param_dict[coords[0]]["('A', 'foo')"]["node_tech"] == 1
        assert param_dict[coords[0]]["('A', 'foo')"]["resource_unit"] == "bar"
        assert param_dict[coords[1]]["('A', 'foo', 'monetary')"]["cost_energy_cap"] == 2

    def test_set_tech_info(self):
        param_dict = AttrDict()
        set_tech_info(param_dict, ["foo"], "resource", 1)
        assert set(param_dict.keys()) == set(["('techs',)"])
        assert param_dict["('techs',)"]["('foo',)"]["resource"] == 1

    def test_add_attributes(self):
        model_run = AttrDict({"applied_overrides": "foo", "scenario": "bar"})
        attr_dict = add_attributes(model_run)
        assert set(attr_dict.keys()) == set(
            ["calliope_version", "applied_overrides", "scenario", "defaults"]
        )
        attr_dict["calliope_version"] == __version__
        assert attr_dict["applied_overrides"] == "foo"
        assert attr_dict["scenario"] == "bar"
        assert "\ncost_energy_cap" in attr_dict["defaults"]
        assert "\nenergy_cap_max" in attr_dict["defaults"]
        assert "\navailable_area" in attr_dict["defaults"]
