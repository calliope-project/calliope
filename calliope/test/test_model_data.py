import pytest
import os
import logging

import numpy as np
import pandas as pd

import calliope
from calliope.preprocess.model_data import ModelDataFactory
from calliope.core.attrdict import AttrDict
from calliope._version import __version__
import calliope.exceptions as exceptions

from calliope.preprocess import model_run_from_yaml
from calliope.test.common.util import check_error_or_warning


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
        return ModelDataFactory(model_run)

    def test_model_data_init(self, model_data):
        for attr in [
            "LOOKUP_STR",
            "UNWANTED_TECH_KEYS",
            "node_dict",
            "tech_dict",
            "model_data",
            "template_config",
            "link_techs",
        ]:
            assert hasattr(model_data, attr)

        for var in ["node_tech", "link_remote_techs", "link_remote_nodes"]:
            assert var in model_data.model_data.data_vars.keys()

        assert model_data.model_data.attrs != {}

    def test_model_data_init_vars(self, model_data):
        non_link_node_techs = [
            ("a", "test_demand_elec"),
            ("a", "test_supply_elec"),
            ("b", "test_demand_elec"),
            ("b", "test_supply_elec"),
        ]
        assert len(model_data.model_data.node_tech.to_series().dropna()) == 8
        assert len(model_data.model_data.link_remote_techs.to_series().dropna()) == 4
        assert len(model_data.model_data.link_remote_nodes.to_series().dropna()) == 4
        assert (
            model_data.model_data.node_tech.to_series()
            .dropna()
            .index.difference(
                model_data.model_data.link_remote_techs.to_series().dropna().index
            )
            .difference(non_link_node_techs)
            .empty
        )
        assert (
            model_data.model_data.link_remote_nodes.to_series()
            .dropna()
            .index.difference(
                model_data.model_data.link_remote_techs.to_series().dropna().index
            )
            .empty
        )

    @pytest.mark.parametrize(
        ("var", "invalid"),
        [
            (None, True),
            ([], True),
            ((), True),
            (set(), True),
            (dict(), True),
            (np.nan, True),
            ("foo", False),
            (1, False),
            (0, False),
            ([1], False),
            ((1,), False),
            ({1}, False),
            ({"foo": "bar"}, False),
        ],
    )
    def test_empty_or_invalid(self, model_data, var, invalid):
        assert model_data._empty_or_invalid(var) == invalid

    def test_strip_unwanted_keys(self, model_data, model_run):
        model_data.tech_dict = model_run.techs.as_dict_flat()
        model_data.node_dict = model_run.nodes.as_dict_flat()
        initial_node_length = len(model_data.node_dict)
        assert all(
            any(tech_info.endswith(key) for tech_info in model_data.tech_dict.keys())
            for key in model_data.UNWANTED_TECH_KEYS
        )
        model_data._strip_unwanted_keys()
        assert initial_node_length == len(model_data.node_dict)
        assert not any(
            any(tech_info.endswith(key) for tech_info in model_data.tech_dict.keys())
            for key in model_data.UNWANTED_TECH_KEYS
        )
        assert len(model_data.stripped_keys) > 0

    @pytest.mark.parametrize(
        ("initial_string", "result"),
        [("{}", "[\\w\\-]*"), ("{0}", "[\\w\\-]*"), ("{0}{0}", "[\\w\\-]*[\\w\\-]*")],
    )
    def test_format_lookup(self, model_data, initial_string, result):
        assert model_data._format_lookup(initial_string) == result

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
        matching_search_results = model_data._get_key_matching_nesting(
            nesting, key_to_check
        )
        if search_result is not None:
            assert matching_search_results.groups() == tuple(search_result)
        else:
            assert matching_search_results is None

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
        matching_search_results = model_data._get_key_matching_nesting(
            nesting, key_to_check, start, end
        )
        if search_result is not None:
            assert matching_search_results.groups() == tuple(search_result)
        else:
            assert matching_search_results is None

    @pytest.mark.parametrize(
        ("model_run_dict", "nesting", "expected_data_dict"),
        [
            (
                {"A.techs.B.constraints.C": "D"},
                ["techs", "constraints"],
                {("A", "B", "C"): "D"},
            ),
            ({"A.techs.B.constraints.C": 2}, ["techs", "?{0}"], {("A", "B", "C"): 2}),
            ({"A.techs.C": ["a", "b"]}, ["techs"], {("A", "C"): "a.b"}),
            ({"A.techs.C": 2, "A.foo.C": 2}, ["techs"], {("A", "C"): 2}),
            ({"A.techs.C": 2, "A.foo.C": 3}, ["foo"], {("A", "C"): 3}),
        ],
    )
    @pytest.mark.parametrize("get_method", ["get", "pop"])
    def test_reformat_model_run_dict(
        self, model_data, model_run_dict, nesting, expected_data_dict, get_method
    ):
        init_model_run_dict = model_run_dict.copy()
        data_dict = model_data._reformat_model_run_dict(
            model_run_dict, nesting, get_method
        )
        assert data_dict == expected_data_dict
        if get_method == "pop":
            assert len(model_run_dict) == len(init_model_run_dict) - 1
        elif get_method == "get":
            assert model_run_dict == init_model_run_dict

    @pytest.mark.parametrize(
        ("model_run_dict", "nesting", "expected_data_dict"),
        [
            (
                {"A.techs.B.constraints.C": "D"},
                ["techs", "constraints"],
                {("A", "B", "C", "D"): 1},
            ),
            (
                {"A.techs.C": ["D", "E"]},
                ["techs"],
                {("A", "C", "D"): 1, ("A", "C", "E"): 1},
            ),
        ],
    )
    def test_reformat_model_run_dict_values_as_dim(
        self, model_data, model_run_dict, nesting, expected_data_dict
    ):
        data_dict = model_data._reformat_model_run_dict(
            model_run_dict, nesting, get_method="get", values_as_dimension=True
        )
        assert data_dict == expected_data_dict

    def test_reformat_model_run_dict_no_match(self, model_data):
        data_dict = model_data._reformat_model_run_dict(
            {"A.techs.B": 2}, ["foo"], get_method="get"
        )
        assert data_dict is None

    def test_dict_to_df_basic(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "b"]
        df = model_data._dict_to_df(data_dict, dims)
        assert df.index.names == dims
        assert df.index[0] == list(data_dict.keys())[0][:-1]
        assert df.columns[0] == list(data_dict.keys())[0][-1]
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_var_name(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "b", "c"]
        df = model_data._dict_to_df(data_dict, dims, var_name="foo")
        assert df.index.names == dims
        assert df.index[0] == list(data_dict.keys())[0]
        assert df.columns[0] == "foo"
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_var_name_in_dims(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "var_name", "c"]
        df = model_data._dict_to_df(data_dict, dims)
        assert df.index.names == ("a", "c")
        assert df.index[0] == ("A", "C")
        assert df.columns[0] == "B"
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_var_name_prefix(self, model_data):
        data_dict = {("A", "B", "C"): 1}
        dims = ["a", "b"]
        df = model_data._dict_to_df(data_dict, dims, var_name_prefix="foo")
        assert df.index.names == dims
        assert df.index[0] == list(data_dict.keys())[0][:-1]
        assert df.columns[0] == "foo_" + list(data_dict.keys())[0][-1]
        assert df.values[0] == list(data_dict.values())[0]

    def test_dict_to_df_is_link(self, model_data):
        data_dict = {("A", "B", "C", "D"): 1}
        dims = ["techs", "nodes", "node_to"]
        df = model_data._dict_to_df(data_dict, dims, is_link=True)
        assert df.index.names == ("nodes", "techs")
        assert df.index[0] == ("B", "A:C")
        assert df.columns[0] == list(data_dict.keys())[0][-1]
        assert df.values[0] == list(data_dict.values())[0]

    def test_model_run_dict_to_dataset_no_match(self, caplog, model_data):
        with caplog.at_level(logging.INFO):
            model_data._model_run_dict_to_dataset(
                "foo", "node", ["foobar"], ["nodes", "foobar"]
            )
        records = [r.msg for r in caplog.records]
        assert "No relevant data found for `foo` group of parameters" in records

    @pytest.mark.parametrize(
        ("data", "idx", "cols", "out_idx"),
        [
            (
                ["foo"],
                ["test_transmission_elec"],
                ["foobar"],
                ["test_transmission_elec:a", "test_transmission_elec:b"],
            ),
            (
                ["foo", "bar"],
                ["test_transmission_elec", "test_transmission_heat"],
                ["foobar"],
                [
                    "test_transmission_elec:a",
                    "test_transmission_elec:b",
                    "test_transmission_heat:a",
                    "test_transmission_heat:b",
                ],
            ),
            (["foo"], ["bar"], ["foobar"], ["bar"]),
        ],
    )
    def test_update_link_tech_names(self, model_data, data, idx, cols, out_idx):
        df = pd.DataFrame(data=data, index=idx, columns=cols)
        new_df = model_data._update_link_tech_names(df)
        assert new_df.index.difference(out_idx).empty

    @pytest.mark.parametrize(
        ("data", "idx", "cols", "out_idx"),
        [
            (
                ["foo"],
                [("test_transmission_elec", "elec")],
                ["foobar"],
                [
                    ("test_transmission_elec:a", "elec"),
                    ("test_transmission_elec:b", "elec"),
                ],
            ),
            (["foo"], [("bar", "baz")], ["foobar"], [("bar", "baz")]),
        ],
    )
    def test_update_link_tech_names_multiindex(
        self, model_data, data, idx, cols, out_idx
    ):
        multiindex = pd.MultiIndex.from_tuples(idx, names=["techs", "blah"])
        df = pd.DataFrame(data=data, index=multiindex, columns=cols)
        new_df = model_data._update_link_tech_names(df)
        assert new_df.index.difference(out_idx).empty

    def test_update_link_idx_levels(self, model_data):
        idx = pd.MultiIndex.from_tuples(
            [("foo", "bar", "baz", "blah"), ("foo1", "bar1", "baz1", "blah")],
            names=["techs", "node_to", "nodes", "blah"],
        )
        df = pd.DataFrame(data=[1, 2], index=idx, columns=["foobar"])
        new_df = model_data._update_link_idx_levels(df)
        assert new_df.index.names == ["nodes", "blah", "techs"]
        assert new_df.index.difference(
            [("baz", "blah", "foo:bar"), ("baz1", "blah", "foo1:bar1")]
        ).empty

    def test_all_df_to_true(self, model_data):
        df = pd.DataFrame(data=["a", "b"], index=["foo", "bar"], columns=["foobar"])
        new_df = model_data._all_df_to_true(df)
        assert new_df.foobar.dtype.kind == "b"
        assert new_df.foobar.sum() == len(new_df)

    def test_extract_node_tech_data(self, model_data, model_run):
        assert not set(model_data.model_data.data_vars.keys()).difference(
            ["node_tech", "link_remote_techs", "link_remote_nodes"]
        )
        model_data_init = model_data.model_data.copy()
        model_data._extract_node_tech_data()
        model_data_new = model_data.model_data
        for data_var in ["node_tech", "link_remote_techs", "link_remote_nodes"]:
            assert model_data_init[data_var].equals(model_data_new[data_var])
        for coord in ["carriers", "carrier_tiers"]:
            assert coord not in model_data_init
            assert coord in model_data.model_data
        for var in model_data_new.data_vars.values():
            assert "timesteps" not in var.dims

        for key in model_run.nodes.as_dict_flat().keys():
            if "constraints" in key or "switches" in key:
                assert key.split(".")[-1] in model_data_new.data_vars.keys()

    def test_add_time_dimension(self, model_data):
        model_data._extract_node_tech_data()
        assert not hasattr(model_data, "data_pre_time")
        assert not hasattr(model_data, "model_data_pre_clustering")

        model_data._add_time_dimension()
        assert hasattr(model_data, "data_pre_time")
        assert hasattr(model_data, "model_data_pre_clustering")

        assert "timesteps" in model_data.model_data.resource.dims
        assert "max_demand_timesteps" in model_data.model_data.data_vars.keys()
        for var in model_data.model_data.data_vars.values():
            assert (
                var.to_series()
                .astype(str)
                .where(lambda x: (x.str.find("file=") > -1) | (x.str.find("df=") > -1))
                .dropna()
                .empty
            )

    def test_clean_model_data(self, model_data):
        model_data._extract_node_tech_data()
        model_data._add_time_dimension()
        for var in model_data.model_data.data_vars.values():
            assert var.attrs == {}

        model_data._clean_model_data()
        for var in model_data.model_data.data_vars.values():
            assert var.attrs == {"parameters": 1, "is_result": 0}

    @pytest.mark.parametrize("subdict", ["tech", "node"])
    def test_check_data(self, model_data, subdict):
        model_data._extract_node_tech_data()
        setattr(model_data, f"{subdict}_dict", {"foo": 1})
        with pytest.raises(exceptions.ModelError) as errmsg:
            model_data._check_data()
        assert check_error_or_warning(
            errmsg, "Some data not extracted from inputs into model dataset"
        )

    def test_add_attributes(self, model_data):
        model_data.model_data.attrs = {}
        model_run = AttrDict({"applied_overrides": "foo", "scenario": "bar"})
        model_data._add_attributes(model_run)
        attr_dict = model_data.model_data.attrs
        assert set(attr_dict.keys()) == set(
            ["calliope_version", "applied_overrides", "scenario", "defaults"]
        )
        attr_dict["calliope_version"] == __version__
        assert attr_dict["applied_overrides"] == "foo"
        assert attr_dict["scenario"] == "bar"
        assert "cost_energy_cap" in attr_dict["defaults"]
        assert "energy_cap_max" in attr_dict["defaults"]
        assert "available_area" in attr_dict["defaults"]
