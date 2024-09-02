import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.preprocess import data_sources, load
from calliope.preprocess.model_data import ModelDataFactory

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


@pytest.fixture
def model_def():
    filepath = Path(__file__).parent / "common" / "test_model" / "model.yaml"
    model_def_dict, model_def_path, _ = load.load_model_definition(
        filepath.as_posix(), scenario="simple_supply,empty_tech_node"
    )
    return model_def_dict, model_def_path


@pytest.fixture
def data_source_list(model_def, init_config):
    model_def_dict, model_def_path = model_def
    return [
        data_sources.DataSource(
            init_config, source_name, source_dict, {}, model_def_path
        )
        for source_name, source_dict in model_def_dict.pop("data_sources", {}).items()
    ]


@pytest.fixture
def init_config(config_defaults, model_def):
    model_def_dict, _ = model_def
    config_defaults.union(model_def_dict.pop("config"), allow_override=True)
    return config_defaults["init"]


@pytest.fixture
def model_data_factory(model_def, init_config, model_defaults):
    model_def_dict, _ = model_def
    return ModelDataFactory(
        init_config, model_def_dict, [], {"foo": "bar"}, {"default": model_defaults}
    )


@pytest.fixture
def model_data_factory_w_params(model_data_factory: ModelDataFactory):
    model_data_factory.add_node_tech_data()
    return model_data_factory


@pytest.fixture
def my_caplog(caplog):
    caplog.set_level(logging.DEBUG, logger="calliope.preprocess")
    return caplog


@pytest.mark.filterwarnings("ignore:(?s).*Converting non-nanosecond precision datetime")
class TestModelData:
    @pytest.fixture(scope="class")
    def simple_da(self):
        data = {("foo", 1): [True, 10], ("foo", 2): [False, 20], ("bar", 1): [True, 30]}
        da = pd.Series(data).rename_axis(index=["foobar", "foobaz"]).to_xarray()
        return da

    @pytest.fixture(scope="class")
    def timeseries_da(self):
        data = {
            ("2005-01-01 00:00", "bar"): [True, 10],
            ("2005-01-01 01:00", "bar"): [False, 20],
        }
        da = pd.Series(data).rename_axis(index=["timesteps", "foobaz"]).to_xarray()
        da.coords["timesteps"] = da.coords["timesteps"].astype("M")
        return da

    def test_model_data_init(self, model_data_factory: ModelDataFactory):
        assert model_data_factory.param_attrs["flow_in_eff"] == {"default": 1.0}

        assert model_data_factory.dataset.attrs == {"foo": "bar"}

    def test_add_node_tech_data(self, model_data_factory_w_params: ModelDataFactory):
        assert set(model_data_factory_w_params.dataset.nodes.values) == {"a", "b", "c"}
        assert set(model_data_factory_w_params.dataset.techs.values) == {
            "test_supply_elec",
            "test_demand_elec",
            "test_link_a_b_elec",
            "test_link_a_b_heat",
        }
        assert set(model_data_factory_w_params.dataset.carriers.values) == {
            "electricity",
            "heat",
        }
        assert set(model_data_factory_w_params.dataset.data_vars.keys()) == {
            "nodes_inheritance",
            "distance",
            "techs_inheritance",
            "name",
            "carrier_out",
            "carrier_in",
            "base_tech",
            "flow_cap_max",
            "source_use_max",
            "flow_out_eff",
        }

    def test_update_time_dimension_and_params(
        self, model_data_factory_w_params: ModelDataFactory, timeseries_da
    ):
        model_data_factory_w_params.dataset["timeseries_da"] = timeseries_da
        model_data_factory_w_params.update_time_dimension_and_params()
        assert "timestep_resolution" in model_data_factory_w_params.dataset.data_vars
        assert "timestep_weights" in model_data_factory_w_params.dataset.data_vars

    def test_clean_data_from_undefined_members(
        self, my_caplog, model_data_factory: ModelDataFactory
    ):
        model_data_factory.dataset["carrier_in"] = (
            pd.Series(
                {
                    ("A", "foo", "c1"): True,
                    ("B", "bar", "c2"): np.nan,
                    ("C", "foo", "c1"): True,
                }
            )
            .rename_axis(index=["nodes", "techs", "carriers"])
            .to_xarray()
        )
        model_data_factory.dataset["carrier_out"] = (
            pd.Series(
                {
                    ("A", "foo", "c2"): True,
                    ("B", "bar", "c1"): np.nan,
                    ("C", "foo", "c2"): True,
                }
            )
            .rename_axis(index=["nodes", "techs", "carriers"])
            .to_xarray()
        )

        model_data_factory.dataset["will_remain"] = (
            pd.Series({"foo": 1, "bar": 2}).rename_axis(index="techs").to_xarray()
        )
        model_data_factory.dataset["will_delete"] = (
            pd.Series({"foo": np.nan, "bar": 2}).rename_axis(index="techs").to_xarray()
        )
        model_data_factory.dataset["will_delete_2"] = (
            pd.Series({("foo", "B"): 2})
            .rename_axis(index=["techs", "nodes"])
            .to_xarray()
        )

        model_data_factory.clean_data_from_undefined_members()

        assert (
            "Deleting techs values as they are not defined anywhere in the model: {'bar'}"
            in my_caplog.text
        )
        assert (
            "Deleting nodes values as they are not defined anywhere in the model: {'B'}"
            in my_caplog.text
        )
        assert (
            "Deleting empty parameters: ['will_delete', 'will_delete_2']"
            in my_caplog.text
        )

        assert "will_delete" not in model_data_factory.dataset
        assert "will_delete_2" not in model_data_factory.dataset
        assert model_data_factory.dataset["will_remain"].item() == 1
        assert set(model_data_factory.dataset.techs.values) == {"foo"}
        assert set(model_data_factory.dataset.nodes.values) == {"A", "C"}
        assert model_data_factory.dataset["definition_matrix"].dtype.kind == "b"

    @pytest.mark.parametrize(
        ("existing_distance", "expected_distance"), [(np.nan, 343.834), (1, 1)]
    )
    def test_add_link_distances_missing_distance(
        self,
        my_caplog,
        model_data_factory_w_params: ModelDataFactory,
        existing_distance,
        expected_distance,
    ):
        model_data_factory_w_params.clean_data_from_undefined_members()
        model_data_factory_w_params.dataset["latitude"] = (
            pd.Series({"a": 51.507222, "b": 48.8567})
            .rename_axis(index="nodes")
            .to_xarray()
        )
        model_data_factory_w_params.dataset["longitude"] = (
            pd.Series({"a": -0.1275, "b": 2.3508})
            .rename_axis(index="nodes")
            .to_xarray()
        )
        model_data_factory_w_params.dataset["distance"] = (
            pd.Series({"test_link_a_b_elec": existing_distance})
            .rename_axis(index="techs")
            .to_xarray()
        )

        model_data_factory_w_params.add_link_distances()
        assert "Any missing link distances automatically computed" in my_caplog.text
        assert model_data_factory_w_params.dataset["distance"].sel(
            techs="test_link_a_b_elec"
        ).item() == pytest.approx(expected_distance)

    @pytest.mark.parametrize(("unit", "expected"), [("m", 343834), ("km", 343.834)])
    def test_add_link_distances_no_da(
        self, my_caplog, model_data_factory_w_params: ModelDataFactory, unit, expected
    ):
        _default_distance_unit = model_data_factory_w_params.config["distance_unit"]
        model_data_factory_w_params.config["distance_unit"] = unit
        model_data_factory_w_params.clean_data_from_undefined_members()
        model_data_factory_w_params.dataset["latitude"] = (
            pd.Series({"A": 51.507222, "B": 48.8567})
            .rename_axis(index="nodes")
            .to_xarray()
        )
        model_data_factory_w_params.dataset["longitude"] = (
            pd.Series({"A": -0.1275, "B": 2.3508})
            .rename_axis(index="nodes")
            .to_xarray()
        )
        del model_data_factory_w_params.dataset["distance"]

        model_data_factory_w_params.add_link_distances()
        model_data_factory_w_params.config["distance_unit"] = _default_distance_unit
        assert "Link distance matrix automatically computed" in my_caplog.text
        assert (
            model_data_factory_w_params.dataset["distance"].dropna("techs")
            == pytest.approx(expected)
        ).all()

    def test_add_link_distances_no_latlon(
        self, my_caplog, model_data_factory_w_params: ModelDataFactory
    ):
        model_data_factory_w_params.clean_data_from_undefined_members()
        model_data_factory_w_params.add_link_distances()
        assert "Link distances will not be computed automatically" in my_caplog.text

    def test_add_colors_no_init_da(
        self, my_caplog, model_data_factory_w_params: ModelDataFactory
    ):
        model_data_factory_w_params.add_colors()
        assert "Building technology color" in my_caplog.text
        np.testing.assert_array_equal(
            model_data_factory_w_params.dataset["color"].values,
            ["#19122b", "#17344c", "#185b48", "#3c7632"],
        )

    def test_add_colors_full_init_da(
        self, my_caplog, model_data_factory_w_params: ModelDataFactory
    ):
        model_data_factory_w_params.dataset["color"] = xr.DataArray(
            ["#123", "#654", "#321", "#456"], dims=("techs",)
        )
        color_da_copy = model_data_factory_w_params.dataset["color"].copy()
        model_data_factory_w_params.add_colors()
        assert "technology color" not in my_caplog.text
        assert model_data_factory_w_params.dataset["color"].equals(color_da_copy)

    def test_add_colors_partial_init_da(
        self, my_caplog, model_data_factory_w_params: ModelDataFactory
    ):
        model_data_factory_w_params.dataset["color"] = pd.Series(
            ["#123", np.nan, "#321", "#456"],
            index=model_data_factory_w_params.dataset.techs.to_index(),
        ).to_xarray()

        model_data_factory_w_params.add_colors()
        assert "Filling missing technology color" in my_caplog.text
        np.testing.assert_array_equal(
            model_data_factory_w_params.dataset["color"].values,
            ["#123", "#17344c", "#321", "#456"],
        )

    def test_assign_input_attr(
        self, model_data_factory: ModelDataFactory, simple_da, timeseries_da
    ):
        model_data_factory.dataset["storage_cap_max"] = simple_da
        model_data_factory.dataset["bar"] = timeseries_da
        assert model_data_factory.dataset.data_vars
        assert not any(
            "is_result" in var.attrs
            for var in model_data_factory.dataset.data_vars.values()
        )

        model_data_factory.assign_input_attr()

        assert all(
            var.attrs["is_result"] is False
            for var in model_data_factory.dataset.data_vars.values()
        )
        assert model_data_factory.dataset["storage_cap_max"].attrs["default"] == np.inf
        assert "default" not in model_data_factory.dataset["bar"].attrs

    def test_get_relevant_node_refs_ts_data(self, model_data_factory: ModelDataFactory):
        techs_dict = AttrDict(
            {
                "foo": {
                    "key3": {"data": None, "index": [["foo"]], "dims": ["foobar"]},
                    "key4": 1,
                    "key5": "foobar",
                },
                "bar": None,
            }
        )
        expected_tech_dict = AttrDict(
            {
                "foo": {
                    "key3": {"data": None, "index": [["foo"]], "dims": ["foobar"]},
                    "key4": 1,
                    "key5": "foobar",
                },
                "bar": None,
            }
        )
        model_data_factory._get_relevant_node_refs(techs_dict, "A")
        assert techs_dict == expected_tech_dict

    def test_get_relevant_node_refs_no_ts_data(
        self, model_data_factory: ModelDataFactory
    ):
        techs_dict = AttrDict(
            {
                "foo": {
                    "key1": 1,
                    "key2": {"data": 1},
                    "key3": {"data": 1, "index": [["foo"]], "dims": ["foobar"]},
                },
                "bar": None,
            }
        )
        refs = model_data_factory._get_relevant_node_refs(techs_dict, "A")
        assert set(refs) == set(["key1", "key2", "key3"])

    def test_get_relevant_node_refs_parent_at_node_not_supported(
        self, model_data_factory: ModelDataFactory
    ):
        techs_dict = AttrDict(
            {"bar": {"key1": 1}, "foo": {"base_tech": "foobar"}, "baz": None}
        )
        with pytest.raises(exceptions.ModelError) as excinfo:
            model_data_factory._get_relevant_node_refs(techs_dict, "A")

        assert check_error_or_warning(
            excinfo,
            "(nodes, A), (techs, foo) | Defining a technology `base_tech` at a node is not supported",
        )

    @pytest.mark.parametrize(
        ("param_data", "expected_da"),
        [
            ({"data": 1, "index": [[]], "dims": []}, xr.DataArray(1)),
            (
                {"data": 1, "index": [["foobar"]], "dims": ["foo"]},
                pd.Series({"foobar": 1}).rename_axis(index="foo").to_xarray(),
            ),
        ],
    )
    def test_param_dict_to_array(
        self, model_data_factory: ModelDataFactory, param_data, expected_da
    ):
        da = model_data_factory._param_dict_to_array("foo", param_data)
        assert da.equals(expected_da)

    def test_definition_dict_to_ds(self, model_data_factory: ModelDataFactory):
        def_dict = {
            "test_idx": {
                "foo": 1,
                "bar": {"data": True, "index": "foobaz", "dims": "foobar"},
            }
        }
        dim_name = "test_dim"
        param_ds = model_data_factory._definition_dict_to_ds(def_dict, dim_name)
        expected_ds = xr.Dataset(
            {
                "foo": pd.Series({"test_idx": 1}).rename_axis(index="test_dim"),
                "bar": pd.Series({("test_idx", "foobaz"): True})
                .rename_axis(index=["test_dim", "foobar"])
                .to_xarray(),
            }
        )
        assert param_ds.broadcast_equals(expected_ds)

    @pytest.mark.parametrize(
        ("input_idx", "expected_idx"),
        [
            ("foo", [["foo"]]),
            (["foo"], [["foo"]]),
            ([["foo"]], [["foo"]]),
            (["foo", "bar"], [["foo"], ["bar"]]),
            ([["foo", "bar"], ["foo", "baz"]], [["foo", "bar"], ["foo", "baz"]]),
        ],
    )
    def test_prepare_param_dict_indexed_idx(
        self, model_data_factory: ModelDataFactory, input_idx, expected_idx
    ):
        dict_skeleton = {"data": 1, "dims": ["foo"]}
        output = model_data_factory._prepare_param_dict(
            "foo", {"index": input_idx, **dict_skeleton}
        )
        assert output == {"index": expected_idx, **dict_skeleton}

    @pytest.mark.parametrize(
        ("input_dim", "expected_dim"),
        [("foo", ["foo"]), (["foo"], ["foo"]), (["foo", "bar"], ["foo", "bar"])],
    )
    def test_prepare_param_dict_indexed_dim(
        self, model_data_factory: ModelDataFactory, input_dim, expected_dim
    ):
        dict_skeleton = {"data": 1, "index": [["foo"]]}
        output = model_data_factory._prepare_param_dict(
            "foo", {"dims": input_dim, **dict_skeleton}
        )
        assert output == {"dims": expected_dim, **dict_skeleton}

    def test_prepare_param_dict_unindexed(self, model_data_factory: ModelDataFactory):
        output = model_data_factory._prepare_param_dict("foo", 1)
        assert output == {"data": 1, "index": [[]], "dims": []}

    def test_prepare_param_dict_lookup(
        self, model_data_factory: ModelDataFactory, simple_da: xr.DataArray
    ):
        model_data_factory.LOOKUP_PARAMS["lookup_arr"] = "foobar"
        model_data_factory.dataset["orig"] = simple_da
        output = model_data_factory._prepare_param_dict("lookup_arr", ["foo", "bar"])
        assert output == {"data": True, "index": [["foo"], ["bar"]], "dims": ["foobar"]}

    def test_prepare_param_dict_not_lookup(self, model_data_factory: ModelDataFactory):
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            model_data_factory._prepare_param_dict("foo", ["foo", "bar"])
        assert check_error_or_warning(
            excinfo,
            "foo | Cannot pass parameter data as a list unless the parameter is one of the pre-defined lookup arrays",
        )

    def test_template_defs_inactive(
        self, my_caplog, model_data_factory: ModelDataFactory
    ):
        def_dict = {"A": {"active": False}}
        new_def_dict = model_data_factory._inherit_defs(
            dim_name="nodes", dim_dict=AttrDict(def_dict)
        )
        assert "(nodes, A) | Deactivated." in my_caplog.text
        assert not new_def_dict

    def test_template_defs_nodes_inherit(self, model_data_factory: ModelDataFactory):
        def_dict = {
            "A": {"template": "init_nodes", "my_param": 1},
            "B": {"my_param": 2},
        }
        new_def_dict = model_data_factory._inherit_defs(
            dim_name="nodes", dim_dict=AttrDict(def_dict)
        )

        assert new_def_dict == {
            "A": {
                "nodes_inheritance": "init_nodes",
                "my_param": 1,
                "techs": {"test_demand_elec": None},
            },
            "B": {"my_param": 2},
        }

    def test_template_defs_nodes_from_base(self, model_data_factory: ModelDataFactory):
        """Without a `dim_dict` to start off inheritance chaining, the `dim_name` will be used to find keys."""
        new_def_dict = model_data_factory._inherit_defs(dim_name="nodes")
        assert set(new_def_dict.keys()) == {"a", "b", "c"}

    def test_template_defs_techs(self, model_data_factory: ModelDataFactory):
        """`dim_dict` overrides content of base model definition."""
        model_data_factory.model_definition.set_key("techs.foo.base_tech", "supply")
        model_data_factory.model_definition.set_key("techs.foo.my_param", 2)

        def_dict = {"foo": {"my_param": 1}}
        new_def_dict = model_data_factory._inherit_defs(
            dim_name="techs", dim_dict=AttrDict(def_dict)
        )
        assert new_def_dict == {"foo": {"my_param": 1, "base_tech": "supply"}}

    def test_template_defs_techs_inherit(self, model_data_factory: ModelDataFactory):
        """Use of template is tracked in updated definition dictionary (as `techs_inheritance` here)."""
        model_data_factory.model_definition.set_key(
            "techs.foo.template", "test_controller"
        )
        model_data_factory.model_definition.set_key("techs.foo.base_tech", "supply")
        model_data_factory.model_definition.set_key("techs.foo.my_param", 2)

        def_dict = {"foo": {"my_param": 1}}
        new_def_dict = model_data_factory._inherit_defs(
            dim_name="techs", dim_dict=AttrDict(def_dict)
        )
        assert new_def_dict == {
            "foo": {
                "my_param": 1,
                "base_tech": "supply",
                "techs_inheritance": "test_controller",
            }
        }

    def test_template_defs_techs_empty_def(self, model_data_factory: ModelDataFactory):
        """An empty `dim_dict` entry can be handled, by returning the model definition for that entry."""
        model_data_factory.model_definition.set_key("techs.foo.base_tech", "supply")
        model_data_factory.model_definition.set_key("techs.foo.my_param", 2)

        def_dict = {"foo": None}
        new_def_dict = model_data_factory._inherit_defs(
            dim_name="techs", dim_dict=AttrDict(def_dict)
        )
        assert new_def_dict == {"foo": {"my_param": 2, "base_tech": "supply"}}

    def test_template_defs_techs_missing_base_def(
        self, model_data_factory: ModelDataFactory
    ):
        """If inheriting from a template, checks against the schema will still be undertaken."""
        def_dict = {"foo": {"base_tech": "supply"}}
        with pytest.raises(KeyError) as excinfo:
            model_data_factory._inherit_defs(
                dim_name="techs", dim_dict=AttrDict(def_dict), foobar="bar"
            )
        assert check_error_or_warning(
            excinfo,
            "(foobar, bar), (techs, foo) | Reference to item not defined in base techs",
        )

    @pytest.mark.parametrize(
        ("node_dict", "expected_dict", "expected_inheritance"),
        [
            ({"my_param": 1}, {"my_param": 1}, None),
            (
                {"template": "foo_group"},
                {"my_param": 1, "my_other_param": 2, "template": "foo_group"},
                ["bar_group", "foo_group"],
            ),
            (
                {"template": "bar_group"},
                {"my_param": 2, "my_other_param": 2, "template": "bar_group"},
                ["bar_group"],
            ),
            (
                {"template": "bar_group", "my_param": 3, "my_own_param": 1},
                {
                    "my_param": 3,
                    "my_other_param": 2,
                    "my_own_param": 1,
                    "template": "bar_group",
                },
                ["bar_group"],
            ),
        ],
    )
    def test_climb_template_tree(
        self,
        model_data_factory: ModelDataFactory,
        node_dict,
        expected_dict,
        expected_inheritance,
    ):
        """Templates should be found and applied in order of 'ancestry' (newer dict keys replace older ones if they overlap)."""
        group_dict = {
            "foo_group": {"template": "bar_group", "my_param": 1},
            "bar_group": {"my_param": 2, "my_other_param": 2},
        }
        model_data_factory.model_definition["templates"] = AttrDict(group_dict)
        new_dict, inheritance = model_data_factory._climb_template_tree(
            AttrDict(node_dict), "nodes", "A"
        )
        assert new_dict == expected_dict
        assert inheritance == expected_inheritance

    def test_climb_template_tree_missing_ancestor(
        self, model_data_factory: ModelDataFactory
    ):
        """Referencing a template that doesn't exist in `templates` raises an error."""
        group_dict = {
            "foo_group": {"template": "bar_group", "my_param": 1},
            "bar_group": {"my_param": 2, "my_other_param": 2},
        }
        model_data_factory.model_definition["templates"] = AttrDict(group_dict)
        with pytest.raises(KeyError) as excinfo:
            model_data_factory._climb_template_tree(
                AttrDict({"template": "not_there"}), "nodes", "A"
            )

        assert check_error_or_warning(excinfo, "(nodes, A) | Cannot find `not_there`")

    def test_deactivate_single_dim(self, model_data_factory_w_params: ModelDataFactory):
        assert "a" in model_data_factory_w_params.dataset.nodes
        model_data_factory_w_params._deactivate_item(nodes="a")
        assert "a" not in model_data_factory_w_params.dataset.nodes

    def test_deactivate_two_dims(self, model_data_factory_w_params: ModelDataFactory):
        to_drop = {"nodes": "a", "techs": "test_supply_elec"}
        model_data_factory_w_params._deactivate_item(**to_drop)
        assert "a" in model_data_factory_w_params.dataset.nodes
        assert "test_supply_elec" in model_data_factory_w_params.dataset.techs
        assert (
            model_data_factory_w_params.dataset.carrier_in.sel(**to_drop).isnull().all()
        )
        assert (
            model_data_factory_w_params.dataset.carrier_out.sel(**to_drop)
            .isnull()
            .all()
        )

    @pytest.mark.parametrize(
        "to_drop",
        [
            {"nodes": "d"},
            {"techs": "new_tech"},
            {"nodes": "d", "techs": "test_supply_elec"},
            {"nodes": "a", "techs": "new_tech"},
        ],
    )
    def test_deactivate_no_action(
        self, model_data_factory_w_params: ModelDataFactory, to_drop: dict
    ):
        orig_dataset = model_data_factory_w_params.dataset.copy(deep=True)
        model_data_factory_w_params._deactivate_item(**to_drop)
        assert model_data_factory_w_params.dataset.equals(orig_dataset)

    def test_links_to_node_format_all_active(
        self, my_caplog, model_data_factory: ModelDataFactory
    ):
        node_dict = {
            "a": {"foo": {"base_tech": "supply"}},
            "b": {"bar": {"base_tech": "demand"}},
        }
        link_dict = model_data_factory._links_to_node_format(node_dict)
        assert "Deactivated" not in my_caplog.text
        assert set(link_dict.keys()) == {"a", "b"}
        assert all(
            set(subdict.keys()) == {"test_link_a_b_heat", "test_link_a_b_elec"}
            for subdict in link_dict.values()
        )
        assert not any(
            "to" in subdict["test_link_a_b_elec"] for subdict in link_dict.values()
        )
        assert not any(
            "from" in subdict["test_link_a_b_elec"] for subdict in link_dict.values()
        )

    def test_links_to_node_format_none_active(
        self, my_caplog, model_data_factory: ModelDataFactory
    ):
        node_dict = {"c": {"foo": {"base_tech": "supply"}}}
        link_dict = model_data_factory._links_to_node_format(node_dict)
        assert (
            "(links, test_link_a_b_elec) | Deactivated due to missing" in my_caplog.text
        )
        assert not link_dict

    def test_links_to_node_format_one_active(
        self, my_caplog, model_data_factory: ModelDataFactory
    ):
        node_dict = {
            "a": {"foo": {"base_tech": "supply"}},
            "c": {"bar": {"base_tech": "demand"}},
        }
        link_dict = model_data_factory._links_to_node_format(node_dict)
        assert (
            "(links, test_link_a_b_elec) | Deactivated due to missing" in my_caplog.text
        )
        assert not link_dict

    def test_links_to_node_format_one_way(self, model_data_factory: ModelDataFactory):
        model_data_factory.model_definition["techs"]["test_link_a_b_elec"][
            "one_way"
        ] = True
        node_dict = {
            "a": {"foo": {"base_tech": "supply"}},
            "b": {"bar": {"base_tech": "demand"}},
        }
        link_dict = model_data_factory._links_to_node_format(node_dict)
        assert "carrier_out" not in link_dict["a"]["test_link_a_b_elec"]
        assert "carrier_in" not in link_dict["b"]["test_link_a_b_elec"]

        assert "carrier_in" in link_dict["a"]["test_link_a_b_elec"]
        assert "carrier_out" in link_dict["b"]["test_link_a_b_elec"]

        assert (
            f"carrier_{j}" in link_dict[node]["test_link_a_b_heat"]
            for node in ["a", "b"]
            for j in ["in", "out"]
        )

    @pytest.mark.parametrize("coord_name", ["foosteps", "barsteps"])
    def test_add_to_dataset_timeseries(
        self, my_caplog, model_data_factory: ModelDataFactory, coord_name
    ):
        new_idx = pd.Index(["2005-01-01 00:00", "2005-01-01 01:00"], name=coord_name)
        new_param = pd.DataFrame({"ts_data": [True, False]}, index=new_idx).to_xarray()
        model_data_factory._add_to_dataset(new_param, "foo")

        assert (
            f"foo | Updating `{coord_name}` dimension index values to datetime format"
            in my_caplog.text
        )
        assert model_data_factory.dataset.coords[coord_name].dtype.kind == "M"
        assert "ts_data" in model_data_factory.dataset

    def test_add_to_dataset_no_timeseries(
        self, my_caplog, model_data_factory: ModelDataFactory, simple_da: xr.DataArray
    ):
        new_param = simple_da.copy().to_dataset(name="non_ts_data")
        model_data_factory._add_to_dataset(new_param, "foo")

        assert "dimension index values to datetime format" not in my_caplog.text
        # make sure nothing has changed in the array
        assert "non_ts_data" in model_data_factory.dataset
        assert model_data_factory.dataset["non_ts_data"].equals(simple_da)

    @pytest.mark.parametrize(
        ("data", "kind"),
        [
            ([1, 2], "i"),
            (["1", "2"], "i"),
            (["1", 2], "i"),
            ([1, "2"], "i"),
            ([1.0, 2.0], "f"),
            (["1.0", "2.0"], "f"),
            ([1, "2.0"], "f"),
            (["1", 2.0], "f"),
        ],
    )
    def test_update_numeric_dims(
        self, my_caplog, model_data_factory: ModelDataFactory, data, kind
    ):
        new_idx = pd.Index(data, name="bar")
        new_param = pd.DataFrame({"my_data": [True, False]}, index=new_idx).to_xarray()
        updated_ds = model_data_factory._update_numeric_dims(new_param, "foo")

        assert (
            "foo | Updating `bar` dimension index values to numeric type"
            in my_caplog.text
        )
        assert updated_ds.coords["bar"].dtype.kind == kind

    @pytest.mark.parametrize(("data", "kind"), [(["1", 2], "i"), ([1.0, "2.0"], "f")])
    def test_update_numeric_dims_in_model_data(
        self, my_caplog, model_data_factory: ModelDataFactory, data, kind
    ):
        new_idx = pd.Index(data, name="bar")
        new_param = pd.DataFrame({"num_data": [True, False]}, index=new_idx).to_xarray()
        model_data_factory._add_to_dataset(new_param, "foo")

        assert (
            "foo | Updating `bar` dimension index values to numeric type"
            in my_caplog.text
        )
        assert model_data_factory.dataset.coords["bar"].dtype.kind == kind

    @pytest.mark.parametrize(
        "data", [["foo", 2], [1.0, "foo"], ["foo", "bar"], ["Y1", "Y2"]]
    )
    def test_update_numeric_dims_no_update(
        self, my_caplog, model_data_factory: ModelDataFactory, data
    ):
        new_idx = pd.Index(data, name="bar")
        new_param = pd.DataFrame({"ts_data": [True, False]}, index=new_idx).to_xarray()
        updated_ds = model_data_factory._update_numeric_dims(new_param, "foo")

        assert (
            "foo | Updating `bar` dimension index values to numeric type"
            not in my_caplog.text
        )
        assert updated_ds.coords["bar"].dtype.kind not in ["f", "i"]

    @pytest.mark.parametrize(
        ("coords", "new_coords"),
        [(["foobar", "baz"], ["baz"]), (["bazfoo", "baz"], ["bazfoo", "baz"])],
    )
    def test_log_param_updates_new_coord(
        self,
        my_caplog,
        model_data_factory: ModelDataFactory,
        simple_da: xr.DataArray,
        coords,
        new_coords,
    ):
        model_data_factory.dataset["orig"] = simple_da
        new_param = simple_da.to_series().rename_axis(index=coords).to_xarray()
        model_data_factory._log_param_updates("foo", new_param)
        for coord in new_coords:
            assert (
                f"(parameters, foo) | Adding a new dimension to the model: {coord}"
                in my_caplog.text
            )

    @pytest.mark.parametrize(
        ("index", "new_items"),
        [
            (("hello", 10), [("foobar", "hello")]),
            (("hello", 30), [("foobar", "hello"), ("foobaz", 30)]),
        ],
    )
    def test_log_param_extends_coord(
        self,
        my_caplog,
        model_data_factory: ModelDataFactory,
        simple_da: xr.DataArray,
        index,
        new_items,
    ):
        model_data_factory.dataset["orig"] = simple_da
        new_param = (
            pd.concat([simple_da.to_series(), pd.Series({index: [False, 1]})])
            .rename_axis(index=simple_da.dims)
            .to_xarray()
        )
        model_data_factory._log_param_updates("foo", new_param)
        for item in new_items:
            coord_name, val = item
            val = f"'{val}'" if isinstance(val, str) else val
            assert (
                f"(parameters, foo) | Adding a new value to the `{coord_name}` model coordinate: [{val}]"
                in my_caplog.text
            )

    def test_log_param_no_logging_message(
        self, my_caplog, model_data_factory: ModelDataFactory, simple_da: xr.DataArray
    ):
        model_data_factory.dataset["orig"] = simple_da
        new_param = simple_da.copy()
        model_data_factory._log_param_updates("foo", new_param)

        assert "(parameters, foo) | Adding" not in my_caplog.text

    def test_raise_error_on_transmission_tech_in_node(
        self, model_data_factory: ModelDataFactory
    ):
        tech_def = {
            "tech1": {"base_tech": "supply"},
            **{
                f"tech{num}": {"base_tech": "transmission", "other_param": 1}
                for num in [2, 3]
            },
        }
        with pytest.raises(exceptions.ModelError) as excinfo:
            model_data_factory._raise_error_on_transmission_tech_def(
                AttrDict(tech_def), "foo"
            )
        assert check_error_or_warning(
            excinfo,
            "(nodes, foo) | Transmission techs cannot be directly defined at nodes; they will be automatically assigned to nodes based on `to` and `from` parameters: ['tech2', 'tech3']",
        )


class TestTopLevelParams:
    @pytest.fixture
    def run_and_test(self, model_data_factory_w_params):
        def _run_and_test(in_dict, out_dict, dims):
            model_data_factory_w_params.model_definition["parameters"] = {
                "my_val": in_dict
            }
            model_data_factory_w_params.add_top_level_params()

            _data = pd.Series(out_dict).rename_axis(index=dims)
            pd.testing.assert_series_equal(
                model_data_factory_w_params.dataset.my_val.to_series()
                .dropna()
                .reindex(_data.index),
                _data,
                check_dtype=False,
                check_names=False,
                check_exact=False,
            )

        return _run_and_test

    def test_parameter_already_exists(self):
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model({"parameters.flow_out_eff": 1}, "simple_supply,two_hours")
        assert check_error_or_warning(
            excinfo,
            "A parameter with this name has already been defined in a data source or at a node/tech level.",
        )

    @pytest.mark.parametrize("val", [1, 1.0, np.inf, "foo"])
    def test_top_level_param_single_val(self, val):
        model = build_model({"parameters.my_val": val}, "simple_supply,two_hours")
        assert model.inputs.my_val == xr.DataArray(val)

    @pytest.mark.parametrize("val", [None, np.nan])
    def test_top_level_param_single_val_cleaned_out_in_preprocessing(self, val):
        model = build_model({"parameters.my_val": val}, "simple_supply,two_hours")
        assert "my_val" not in model.inputs

    @pytest.mark.parametrize("val", [1, 1.0, np.inf, "foo"])
    def test_top_level_param_single_data_single_known_dim(self, val, run_and_test):
        run_and_test(
            {"data": val, "index": ["test_supply_elec"], "dims": "techs"},
            {"test_supply_elec": val},
            "techs",
        )

    def test_top_level_param_multi_data_single_known_dim(self, run_and_test):
        run_and_test(
            {
                "data": [1, "foo"],
                "index": ["test_supply_elec", "test_demand_elec"],
                "dims": "techs",
            },
            {"test_supply_elec": 1, "test_demand_elec": "foo"},
            "techs",
        )

    def test_top_level_param_single_data_multi_known_dim(self, run_and_test):
        run_and_test(
            {
                "data": 10,
                "index": [
                    ["electricity", "test_supply_elec"],
                    ["electricity", "test_demand_elec"],
                ],
                "dims": ["carriers", "techs"],
            },
            {
                ("electricity", "test_supply_elec"): 10,
                ("electricity", "test_demand_elec"): 10,
            },
            ["carriers", "techs"],
        )

    def test_top_level_param_multi_data_multi_known_dim(self, run_and_test):
        with pytest.warns(exceptions.ModelWarning) as warninfo:
            run_and_test(
                {
                    "data": [10, 20],
                    "index": [["a", "test_supply_elec"], ["b", "test_demand_elec"]],
                    "dims": ["nodes", "techs"],
                },
                {("a", "test_supply_elec"): 10, ("b", "test_demand_elec"): 20},
                ["nodes", "techs"],
            )
        assert check_error_or_warning(
            warninfo,
            "This parameter will only take effect if you have already defined the following combinations of techs at nodes in your model definition: [('a', 'test_supply_elec') ('b', 'test_demand_elec')]",
        )

    def test_top_level_param_unknown_dim_only(self, my_caplog, run_and_test):
        run_and_test({"data": 10, "index": ["foo"], "dims": "bar"}, {"foo": 10}, "bar")
        assert (
            "(parameters, my_val) | Adding a new dimension to the model: bar"
            in my_caplog.text
        )

    def test_top_level_param_multi_unknown_dim(self, my_caplog, run_and_test):
        run_and_test(
            {"data": 10, "index": [["foo", "foobar"]], "dims": ["bar", "baz"]},
            {("foo", "foobar"): 10},
            ["bar", "baz"],
        )
        assert (
            "(parameters, my_val) | Adding a new dimension to the model: bar"
            in my_caplog.text
        )
        assert (
            "(parameters, my_val) | Adding a new dimension to the model: baz"
            in my_caplog.text
        )

    def test_top_level_param_unknown_dim_mixed(self, my_caplog, run_and_test):
        run_and_test(
            {
                "data": 10,
                "index": [["test_supply_elec", "foobar"]],
                "dims": ["techs", "baz"],
            },
            {("test_supply_elec", "foobar"): 10},
            ["techs", "baz"],
        )
        assert (
            "(parameters, my_val) | Adding a new dimension to the model: baz"
            in my_caplog.text
        )

    def test_top_level_param_timeseries(self, my_caplog, run_and_test):
        run_and_test(
            {"data": 10, "index": ["2005-01-01"], "dims": ["timesteps"]},
            {pd.to_datetime("2005-01-01"): 10},
            "timesteps",
        )
        assert (
            "(parameters, my_val) | Updating `timesteps` dimension index values to datetime format"
            in my_caplog.text
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Operational mode requires the same timestep resolution:calliope.exceptions.ModelWarning"
    )
    def test_top_level_param_extend_dim_vals(
        self, my_caplog, run_and_test, model_data_factory_w_params
    ):
        # We do this test with timesteps as all other dimension elements are filtered out if there is no matching True element in `definition_matrix`
        run_and_test(
            {"data": 10, "index": ["d"], "dims": ["nodes"]}, {"d": 10}, "nodes"
        )
        assert (
            "(parameters, my_val) | Adding a new value to the `nodes` model coordinate: ['d']"
            in my_caplog.text
        )


class TestActiveFalse:
    """Test removal of techs, nodes, links, and transmission techs
    with the ``active: False`` configuration option.

    """

    def test_tech_active_false(self, my_caplog):
        overrides = {"techs.test_storage.active": False}

        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert "test_storage" not in model._model_data.coords["techs"].values

        # Ensure warnings were raised
        assert "(techs, test_storage) | Deactivated" in my_caplog.text

    def test_node_active_false(self, my_caplog):
        overrides = {"nodes.b.active": False}

        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert "b" not in model._model_data.coords["nodes"].values

        # Ensure warnings were raised
        assert (
            "(links, test_link_a_b_elec) | Deactivated due to missing/deactivated `from` or `to` node."
            in my_caplog.text
        )
        assert "(nodes, b) | Deactivated." in my_caplog.text

    def test_node_tech_active_false(self, my_caplog):
        overrides = {"nodes.b.techs.test_storage.active": False}
        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert not (
            model._model_data.definition_matrix.sel(
                techs="test_storage", nodes="b"
            ).any(["carriers"])
        )
        assert "(nodes, b), (techs, test_storage) | Deactivated" in my_caplog.text

    def test_link_active_false(self, my_caplog):
        overrides = {"templates.test_transmission.active": False}
        model = build_model(overrides, "simple_storage,two_hours,investment_costs")

        # Ensure what should be gone is gone
        assert not (model._model_data.base_tech == "transmission").any()
        assert "(techs, test_link_a_b_elec) | Deactivated." in my_caplog.text
