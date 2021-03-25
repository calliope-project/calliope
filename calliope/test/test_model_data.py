from re import L, search
import pytest
import os

import calliope
from calliope.preprocess.model_data import ModelData
from calliope.core.attrdict import AttrDict
from calliope._version import __version__
from calliope.test.common.util import build_test_model as build_model

from calliope.preprocess import model_run_from_yaml


@pytest.fixture(scope="module")
def model_run():
    filepath = os.path.join(
        os.path.dirname(calliope.__file__),
        "test",
        "common",
        "test_model",
        "model.yaml",
    )
    return model_run_from_yaml(filepath, scenario="simple_supply")[0]


class TestModelData:
    @pytest.fixture(scope="class")
    def model_data(self, model_run):
        return ModelData(model_run)

    def test_model_run_init(self, model_data):
        for attr in [
            "node_dict",
            "tech_dict",
            "model_data",
            "lookup_str",
            "template_config",
            "link_techs",
        ]:
            assert hasattr(model_data, attr)

        for var in ["node_tech", "link_remote_techs", "link_remote_nodes"]:
            assert var in model_data.model_data.data_vars.keys()

        assert model_data.model_data.attrs != {}

    def test_strip_unwanted_keys(self, model_data, model_run):
        model_data.tech_dict = model_run.techs.as_dict_flat()
        model_data.node_dict = model_run.nodes.as_dict_flat()
        initial_node_length = len(model_data.node_dict)
        assert all(
            any(tech_info.endswith(key) for tech_info in model_data.tech_dict.keys())
            for key in model_data.unwanted_tech_keys
        )
        model_data.strip_unwanted_keys()
        assert initial_node_length == len(model_data.node_dict)
        assert not any(
            any(tech_info.endswith(key) for tech_info in model_data.tech_dict.keys())
            for key in model_data.unwanted_tech_keys
        )

    @pytest.mark.parametrize(
        ("initial_string", "result"),
        [("{}", "[\\w\\-]*"), ("{0}", "[\\w\\-]*"), ("{0}{0}", "[\\w\\-]*[\\w\\-]*")],
    )
    def test_format_lookup(self, model_data, initial_string, result):
        assert model_data.format_lookup(initial_string) == result

    @pytest.mark.parametrize(
        ("nesting", "key_to_check", "search_result"),
        [
            (["techs"], "aA.techs.bB", ["aA", "bB"]),
            (["techs", "constraints"], "A.techs.B.constraints.C", ["A", "B", "C"]),
            (["techs", "?{0}"], "A.techs.B.constraints.C", ["A", "B", "C"]),
            (["techs", "?({0})"], "A.techs.B.C.D", ["A", "B", "C", "D"]),
            (["techs", "?{0}"], "A.techs.B.constraints.C.other.D", None),
            (["techs", "constraints"], "A.techs.B_.constraints.C", ["A", "B_", "C"]),
            (["techs", "constraints"], "A1.techs.B1.constraints.C", ["A1", "B1", "C"]),
            (["techs", "con_2"], "A_-1.techs.B-2.con_2.C_D", ["A_-1", "B-2", "C_D"]),
            (["techs"], "A.techs.B.constraints.C", None),
            (["techs.links.B"], "A.techs.links.B.C", ["A", "C"]),
        ],
    )
    def test_get_key_matching_nesting_default_start_end(
        self, model_data, nesting, key_to_check, search_result
    ):
        search_results = model_data.get_key_matching_nesting(nesting, key_to_check)
        if search_result is not None:
            assert search_results.groups() == tuple(search_result)
        else:
            assert search_results == None

    @pytest.mark.parametrize(
        ("start", "end", "search_result"),
        [
            ("({0})\\.", "\\.({0})", ["aA", "bB"]),
            ("({0})", "({0})", None),
            ("[a-z]({0})\\.", "\\.({0})", ["A", "bB"]),
            ("({0})\\.", "\\.techs.({0})", None),
        ],
    )
    def test_get_key_matching_nesting_new_start_end(
        self, model_data, start, end, search_result
    ):
        nesting = ["techs"]
        key_to_check = "aA.techs.bB"
        search_results = model_data.get_key_matching_nesting(
            nesting, key_to_check, start, end
        )
        if search_result is not None:
            assert search_results.groups() == tuple(search_result)
        else:
            assert search_results == None

    @pytest.mark.parametrize(
        ("model_run_dict", "nesting", "expected_data_dict"),
        [
            (
                {"A.techs.B.constraints.C": 1},
                ["techs", "constraints"],
                {("A", "B", "C"): 1},
            ),
            ({"A.techs.B.constraints.C": 1}, ["techs", "?{0}"], {("A", "B", "C"): 1}),
        ],
    )
    @pytest.mark.parametrize("get_method", ["get", "pop"])
    def test_reformat_model_run_dict(
        self, model_data, model_run_dict, nesting, expected_data_dict, get_method
    ):
        init_model_run_dict = model_run_dict.copy()
        data_dict = model_data.reformat_model_run_dict(
            model_run_dict, nesting, get_method
        )
        assert data_dict == expected_data_dict
        if get_method == "pop":
            assert not model_run_dict
        elif get_method == "get":
            assert model_run_dict == init_model_run_dict

    def test_dict_to_df_basic(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "b"]
        df = model_data.dict_to_df(data_dict, dims)
        assert df.index.names == dims
        assert df.index[0] == list(data_dict.keys())[0][:-1]
        assert df.columns[0] == list(data_dict.keys())[0][-1]
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_var_name(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "b", "c"]
        df = model_data.dict_to_df(data_dict, dims, var_name="foo")
        assert df.index.names == dims
        assert df.index[0] == list(data_dict.keys())[0]
        assert df.columns[0] == "foo"
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_var_name_in_dims(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "var_name", "c"]
        df = model_data.dict_to_df(data_dict, dims)
        assert df.index.names == ("a", "c")
        assert df.index[0] == ("A", "C")
        assert df.columns[0] == "B"
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_var_name_prefix(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "b"]
        df = model_data.dict_to_df(data_dict, dims, var_name_prefix="foo")
        assert df.index.names == dims
        assert df.index[0] == list(data_dict.keys())[0][:-1]
        assert df.columns[0] == "foo_" + list(data_dict.keys())[0][-1]
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_is_link(self, model_data):
        data_dict = {("A", "B", "C", "D"): 1}
        dims = ["techs", "nodes", "node_to"]
        df = model_data.dict_to_df(data_dict, dims, is_link=True)
        assert df.index.names == ("nodes", "techs")
        assert df.index[0] == ("B", "A:C")
        assert df.columns[0] == list(data_dict.keys())[0][-1]
        assert df.values[0] == list(data_dict.values())[0]

    def test_model_run_dict_to_dataset(self):
        pass

    def test_update_link_tech_names(self):
        pass

    def test_add_var_attrs(self):
        pass

    def test_update_link_idx_levels(self):
        pass

    def test_add_node_tech_sets(self):
        pass

    def test_all_df_to_true(self):
        pass

    def test_get_link_remotes(self):
        pass

    def test_clean_unused_techs_nodes_and_carriers(self):
        pass

    def test_add_param_from_template(self):
        pass

    def test_add_time_dimension(self):
        pass

    def test_apply_time_clustering(self):
        pass

    def test_reorganise_xarray_dimensions(self):
        pass

    def test_update_dtypes(self):
        pass

    def test_check_data(self):
        pass

    def test_add_attributes(self, model_data):
        model_data.model_data.attrs = {}
        model_run = AttrDict({"applied_overrides": "foo", "scenario": "bar"})
        model_data.add_attributes(model_run)
        attr_dict = model_data.model_data.attrs
        assert set(attr_dict.keys()) == set(
            ["calliope_version", "applied_overrides", "scenario", "defaults"]
        )
        attr_dict["calliope_version"] == __version__
        assert attr_dict["applied_overrides"] == "foo"
        assert attr_dict["scenario"] == "bar"
        assert "\ncost_energy_cap" in attr_dict["defaults"]
        assert "\nenergy_cap_max" in attr_dict["defaults"]
        assert "\navailable_area" in attr_dict["defaults"]
