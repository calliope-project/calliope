import logging

import calliope
import pandas as pd
import pytest
from calliope.preprocess import data_sources
from calliope.util.schema import CONFIG_SCHEMA, extract_from_schema

from .common.util import check_error_or_warning


@pytest.fixture(scope="module")
def init_config():
    return calliope.AttrDict(extract_from_schema(CONFIG_SCHEMA, "default"))["init"]


@pytest.fixture(scope="class")
def data_dir(tmp_path_factory):
    filepath = tmp_path_factory.mktemp("data_sources")
    return filepath


@pytest.fixture(scope="class")
def generate_data_source_dict(data_dir):
    def _generate_data_source_dict(filename, df, rows, columns, **to_csv_kwargs):
        filepath = data_dir / filename
        df.to_csv(filepath, **to_csv_kwargs)
        return {
            "source": filepath.as_posix(),
            "rows": rows,
            "columns": columns,
            "add_dimensions": {"parameters": "test_param"},
        }

    return _generate_data_source_dict


class TestDataSourceUtils:
    @pytest.fixture(scope="class")
    def source_obj(self, init_config, generate_data_source_dict):
        df = pd.Series({"bar": 0, "baz": 1})
        source_dict = generate_data_source_dict(
            "foo.csv", df, rows="test_row", columns=None, header=None
        )
        ds = data_sources.DataSource(init_config, "ds_name", source_dict)
        ds.input["foo"] = ["foobar"]
        return ds

    def test_name(self, source_obj):
        assert source_obj.name == "ds_name"

    def test_raise_error(self, data_dir, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj._raise_error("bar")
        assert check_error_or_warning(excinfo, "(data_sources, ds_name) | bar.")

    def test_log_message(self, caplog, data_dir, source_obj):
        caplog.set_level(logging.INFO)
        source_obj._log("bar", "info")
        assert "(data_sources, ds_name) | bar." in caplog.text

    @pytest.mark.parametrize(
        ["key", "expected"],
        [("rows", ["test_row"]), ("columns", None), ("foo", ["foobar"])],
    )
    def test_listify_if_defined(self, source_obj, key, expected):
        output = source_obj._listify_if_defined(key)
        if expected is None:
            assert output is expected
        else:
            assert output == expected

    @pytest.mark.parametrize(
        ["loaded", "defined"],
        [
            (["foo"], ["foo"]),
            ([None], ["foo"]),
            ([1], ["foo"]),
            ([None, "bar"], ["foo", "bar"]),
            ([None, 1], ["foo", "bar"]),
        ],
    )
    def test_compare_axis_names_passes(self, source_obj, loaded, defined):
        source_obj._compare_axis_names(loaded, defined, "foobar")

    @pytest.mark.parametrize(
        ["loaded", "defined"],
        [
            (["bar"], ["foo"]),
            ([None, "foo"], ["foo", "bar"]),
            (["bar", 1], ["foo", "bar"]),
        ],
    )
    def test_compare_axis_names_fails(self, source_obj, loaded, defined):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj._compare_axis_names(loaded, defined, "foobar")
        assert check_error_or_warning(excinfo, "Trying to set names for foobar")


class TestDataSourceInitOneLevel:
    @pytest.fixture(scope="class")
    def multi_row_no_col_data(self, generate_data_source_dict):
        df = pd.Series({"bar": 0, "baz": 1})
        return df, generate_data_source_dict(
            "multi_row_no_col_file.csv", df, rows="test_row", columns=None, header=None
        )

    @pytest.fixture(scope="class")
    def multi_row_one_col_data(self, generate_data_source_dict):
        df = pd.DataFrame({"foo": {"bar": 0, "baz": 1}})
        return df, generate_data_source_dict(
            "multi_row_one_col_file.csv", df, rows="test_row", columns="test_col"
        )

    @pytest.fixture(scope="class")
    def one_row_multi_col_data(self, generate_data_source_dict):
        df = pd.DataFrame({"foo": {"bar": 0}, "foobar": {"bar": 1}})
        return df, generate_data_source_dict(
            "one_row_multi_col_file.csv", df, rows="test_row", columns="test_col"
        )

    @pytest.fixture(scope="class")
    def multi_row_multi_col_data(self, generate_data_source_dict):
        df = pd.DataFrame(
            {"foo": {"bar": 0, "baz": 10}, "foobar": {"bar": 0, "baz": 20}}
        )
        return df, generate_data_source_dict(
            "multi_row_multi_col_file.csv", df, rows="test_row", columns="test_col"
        )

    def test_multi_row_no_col(self, init_config, multi_row_no_col_data):
        expected_df, source_dict = multi_row_no_col_data
        ds = data_sources.DataSource(init_config, "ds_name", source_dict)
        test_param = ds.dataset["test_param"]
        assert not set(["test_row"]).symmetric_difference(test_param.dims)
        pd.testing.assert_series_equal(
            test_param.to_series(), expected_df, check_names=False
        )

    @pytest.mark.parametrize(
        "data_source_ref",
        [
            "multi_row_one_col_data",
            "one_row_multi_col_data",
            "multi_row_multi_col_data",
        ],
    )
    def test_multi_row_one_col(self, init_config, request, data_source_ref):
        expected_df, source_dict = request.getfixturevalue(data_source_ref)
        ds = data_sources.DataSource(init_config, "ds_name", source_dict)
        test_param = ds.dataset["test_param"]
        assert not set(["test_row", "test_col"]).symmetric_difference(test_param.dims)
        pd.testing.assert_series_equal(
            test_param.to_series(), expected_df.stack(), check_names=False
        )

    @pytest.mark.parametrize(
        "data_source_ref",
        [
            "multi_row_one_col_data",
            "one_row_multi_col_data",
            "multi_row_multi_col_data",
        ],
    )
    def test_load_from_df(self, init_config, request, data_source_ref):
        expected_df, source_dict = request.getfixturevalue(data_source_ref)
        source_dict["source"] = data_source_ref
        ds = data_sources.DataSource(
            init_config,
            "ds_name",
            source_dict,
            data_source_dfs={data_source_ref: expected_df},
        )
        test_param = ds.dataset["test_param"]
        assert not set(["test_row", "test_col"]).symmetric_difference(test_param.dims)
        pd.testing.assert_series_equal(
            test_param.to_series(), expected_df.stack(), check_names=False
        )

    def test_load_from_df_must_be_df(self, init_config, multi_row_no_col_data):
        expected_df, source_dict = multi_row_no_col_data
        source_dict["source"] = "foo"
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            data_sources.DataSource(
                init_config,
                "ds_name",
                source_dict,
                data_source_dfs={"foo": expected_df},
            )
        assert check_error_or_warning(
            excinfo, "Data source must be a pandas DataFrame."
        )


class TestDataSourceInitMultiLevel:
    @pytest.fixture(scope="class")
    def multi_row_no_col_data(self, generate_data_source_dict):
        df = pd.Series({("bar1", "bar2"): 0, ("baz1", "baz2"): 1})
        return df, generate_data_source_dict(
            "multi_row_no_col_file.csv",
            df,
            rows=["test_row1", "test_row2"],
            columns=None,
            header=None,
        )

    @pytest.fixture(scope="class")
    def multi_row_one_col_data(self, generate_data_source_dict):
        df = pd.DataFrame({"foo": {("bar1", "bar2"): 0, ("baz1", "baz2"): 1}})
        return df, generate_data_source_dict(
            "multi_row_one_col_file.csv",
            df,
            rows=["test_row1", "test_row2"],
            columns=["test_col"],
        )

    @pytest.fixture(scope="class")
    def one_row_multi_col_data(self, generate_data_source_dict):
        df = pd.DataFrame(
            {("foo1", "foo2"): {"bar": 0}, ("foobar1", "foobar2"): {"bar": 1}}
        )
        return df, generate_data_source_dict(
            "one_row_multi_col_file.csv",
            df,
            rows=["test_row"],
            columns=["test_col1", "test_col2"],
        )

    @pytest.fixture(scope="class")
    def multi_row_multi_col_data(self, generate_data_source_dict):
        df = pd.DataFrame(
            {
                ("foo1", "foo2"): {("bar1", "bar2"): 0, ("baz1", "baz2"): 10},
                ("foobar1", "foobar2"): {("bar1", "bar2"): 0, ("baz1", "baz2"): 20},
            }
        )
        return df, generate_data_source_dict(
            "multi_row_multi_col_file.csv",
            df,
            rows=["test_row1", "test_row2"],
            columns=["test_col1", "test_col2"],
        )

    def test_multi_row_no_col(self, init_config, multi_row_no_col_data):
        expected_df, source_dict = multi_row_no_col_data
        ds = data_sources.DataSource(init_config, "ds_name", source_dict)
        test_param = ds.dataset["test_param"]
        assert not set(["test_row1", "test_row2"]).symmetric_difference(test_param.dims)
        pd.testing.assert_series_equal(
            test_param.to_series().dropna(),
            expected_df,
            check_names=False,
            check_dtype=False,
        )

    @pytest.mark.parametrize(
        "data_source_ref",
        [
            "multi_row_one_col_data",
            "one_row_multi_col_data",
            "multi_row_multi_col_data",
        ],
    )
    def test_multi_row_one_col(self, init_config, request, data_source_ref):
        expected_df, source_dict = request.getfixturevalue(data_source_ref)
        ds = data_sources.DataSource(init_config, "ds_name", source_dict)
        test_param = ds.dataset["test_param"]
        all_dims = source_dict["rows"] + source_dict["columns"]
        assert not set(all_dims).symmetric_difference(test_param.dims)
        pd.testing.assert_frame_equal(
            test_param.to_series().dropna().unstack(source_dict["columns"]),
            expected_df,
            check_names=False,
            check_dtype=False,
        )


class TestDataSourceSelectDropAdd:
    @pytest.fixture(scope="class")
    def source_obj(self, init_config):
        def _source_obj(**source_dict_kwargs):
            df = pd.DataFrame(
                {
                    "test_param": {
                        ("bar1", "baz1"): 0,
                        ("bar2", "baz2"): 1,
                        ("bar3", "baz3"): 2,
                        ("bar1", "baz4"): 3,
                    }
                }
            )
            source_dict = {
                "source": "df",
                "rows": ["test_row1", "test_row2"],
                "columns": "parameters",
                **source_dict_kwargs,
            }
            ds = data_sources.DataSource(
                init_config, "ds_name", source_dict, data_source_dfs={"df": df}
            )
            return ds

        return _source_obj

    def test_select_keep_one(self, source_obj):
        data_source = source_obj(select={"test_row1": "bar1"})
        expected = pd.Series({("bar1", "baz1"): 0, ("bar1", "baz4"): 3})
        assert data_source.dataset.coords["test_row1"].item() == "bar1"
        pd.testing.assert_series_equal(
            data_source.dataset.test_param.to_series().dropna(),
            expected.sort_index(),
            check_dtype=False,
            check_names=False,
        )

    def test_select_keep_two(self, source_obj):
        data_source = source_obj(select={"test_row1": ["bar1", "bar2"]})
        expected = pd.Series(
            {("bar1", "baz1"): 0, ("bar2", "baz2"): 1, ("bar1", "baz4"): 3}
        )
        assert not set(["bar1", "bar2"]).symmetric_difference(
            data_source.dataset.coords["test_row1"].values
        )
        pd.testing.assert_series_equal(
            data_source.dataset.test_param.to_series().dropna(),
            expected.sort_index(),
            check_dtype=False,
            check_names=False,
        )

    def test_select_drop_one(self, source_obj):
        data_source = source_obj(
            select={"test_row1": "bar2", "test_row2": "baz2"},
            drop=["test_row1", "test_row2"],
        )
        assert not data_source.dataset.dims
        assert data_source.dataset.test_param.item() == 1

    def test_select_drop_two(self, source_obj):
        data_source = source_obj(select={"test_row1": "bar1"}, drop="test_row1")
        expected = pd.Series({"baz1": 0, "baz4": 3})
        assert "test_row1" not in data_source.dataset.dims
        pd.testing.assert_series_equal(
            data_source.dataset.test_param.to_series().dropna(),
            expected.sort_index(),
            check_dtype=False,
            check_names=False,
        )

    def test_drop_one(self, source_obj):
        data_source = source_obj(drop="test_row1")
        expected = pd.Series({"baz1": 0, "baz2": 1, "baz3": 2, "baz4": 3})
        assert "test_row1" not in data_source.dataset.dims
        pd.testing.assert_series_equal(
            data_source.dataset.test_param.to_series().dropna(),
            expected.sort_index(),
            check_dtype=False,
            check_names=False,
        )


class TestDataSourceMalformed:
    @pytest.fixture(scope="class")
    def source_obj(self, init_config):
        def _source_obj(**source_dict_kwargs):
            df = pd.DataFrame(
                {
                    "foo": {
                        ("bar1", "baz1"): 0,
                        ("bar2", "baz2"): 1,
                        ("bar3", "baz3"): 2,
                        ("bar1", "baz4"): 3,
                    }
                }
            )
            source_dict = {
                "source": "df",
                "rows": ["test_row1", "test_row2"],
                **source_dict_kwargs,
            }
            ds = data_sources.DataSource(
                init_config, "ds_name", source_dict, data_source_dfs={"df": df}
            )
            return ds

        return _source_obj

    def test_check_processed_tdf_no_parameters_dim(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj()
        assert check_error_or_warning(excinfo, "The `parameters` dimension must exist")

    def test_check_processed_tdf_duplicated_idx(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj(drop="test_row2", add_dimensions={"parameters": "test_param"})
        assert check_error_or_warning(excinfo, "Duplicate index items found:")

    def test_check_processed_tdf_duplicated_dim_name(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj(add_dimensions={"test_row2": "foo", "parameters": "test_param"})
        assert check_error_or_warning(excinfo, "Duplicate dimension names found:")

    def test_too_many_called_cols(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj(columns=["foo", "bar"])
        assert check_error_or_warning(
            excinfo, "Expected 2 columns levels in loaded data."
        )

    def test_too_few_called_rows(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj(rows=None)
        assert check_error_or_warning(
            excinfo, "Expected a single index level in loaded data."
        )

    def test_check_for_protected_params(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj(add_dimensions={"parameters": "definition_matrix"})
        assert check_error_or_warning(
            excinfo, "`definition_matrix` is a protected array"
        )


class TestDataSourceLookupDictFromParam:
    @pytest.fixture(scope="class")
    def source_obj(self, init_config):
        df = pd.DataFrame(
            {
                "FOO": {("foo1", "bar1"): 1, ("foo1", "bar2"): 1},
                "BAR": {("foo1", "bar1"): 1, ("foo2", "bar2"): 1},
            }
        )
        source_dict = {
            "source": "df",
            "rows": ["techs", "carriers"],
            "columns": "parameters",
        }
        ds = data_sources.DataSource(
            init_config, "ds_name", source_dict, data_source_dfs={"df": df}
        )
        return ds

    @pytest.mark.parametrize(
        ["param", "expected"],
        [
            ("FOO", {"foo1": {"FOO": ["bar1", "bar2"]}}),
            ("BAR", {"foo1": {"BAR": "bar1"}, "foo2": {"BAR": "bar2"}}),
        ],
    )
    def test_carrier_info_dict_from_model_data_var(self, source_obj, param, expected):
        carrier_info = source_obj.lookup_dict_from_param(param, "carriers")
        assert carrier_info == expected

    def test_carrier_info_dict_from_model_data_var_missing_dim(self, source_obj):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj.lookup_dict_from_param("FOO", "foobar")
        check_error_or_warning(
            excinfo,
            "Loading FOO with missing dimension(s). Must contain `techs` and `foobar`, received: ('techs', 'carriers')",
        )


class TestDataSourceTechDict:
    @pytest.fixture(scope="class")
    def source_obj(self, init_config):
        def _source_obj(df_dict, rows="techs"):
            df = pd.DataFrame(df_dict)
            source_dict = {"source": "df", "rows": rows, "columns": "parameters"}
            ds = data_sources.DataSource(
                init_config, "ds_name", source_dict, data_source_dfs={"df": df}
            )
            return ds

        return _source_obj

    def test_tech_dict_from_one_param(self, source_obj):
        df_dict = {"test_param": {"foo1": 1, "foo2": 2}}
        tech_dict, base_dict = source_obj(df_dict).tech_dict()

        assert tech_dict == {"foo1": {}, "foo2": {}}
        assert base_dict == {}

    def test_tech_dict_from_two_param(self, source_obj):
        df_dict = {"foo": {"foo1": 1, "foo2": 2}, "bar": {"bar1": 1, "bar2": 2}}
        tech_dict, base_dict = source_obj(df_dict).tech_dict()

        assert tech_dict == {"foo1": {}, "foo2": {}, "bar1": {}, "bar2": {}}
        assert base_dict == {}

    def test_tech_dict_from_parent(self, source_obj):
        df_dict = {"base_tech": {"foo1": "transmission", "foo2": "supply"}}
        tech_dict, base_dict = source_obj(df_dict).tech_dict()

        assert tech_dict == {"foo1": {}, "foo2": {}}
        assert base_dict == {
            "foo1": {"base_tech": "transmission"},
            "foo2": {"base_tech": "supply"},
        }

    def test_tech_dict_from_parent_and_param(self, source_obj):
        df_dict = {"base_tech": {"foo1": "transmission"}, "other_param": {"bar1": 1}}
        tech_dict, base_dict = source_obj(df_dict).tech_dict()

        assert tech_dict == {"foo1": {}, "bar1": {}}
        assert base_dict == {"foo1": {"base_tech": "transmission"}}

    def test_tech_dict_from_to_from(self, source_obj):
        df_dict = {
            "from": {"foo1": "bar1", "foo2": "bar2"},
            "to": {"foo1": "bar2", "foo3": "bar1"},
        }
        tech_dict, base_dict = source_obj(df_dict).tech_dict()

        assert tech_dict == {"foo1": {}, "foo2": {}, "foo3": {}}
        assert base_dict == {
            "foo1": {"from": "bar1", "to": "bar2"},
            "foo2": {"from": "bar2"},
            "foo3": {"to": "bar1"},
        }

    def test_tech_dict_empty(self, source_obj):
        df_dict = {"available_area": {"foo1": 1}}
        tech_dict, base_dict = source_obj(df_dict, rows="nodes").tech_dict()

        assert not tech_dict
        assert not base_dict


class TestDataSourceNodeDict:
    @pytest.fixture(scope="class")
    def source_obj(self, init_config):
        def _source_obj(df_dict, rows=["nodes", "techs"]):
            df = pd.DataFrame(df_dict)
            source_dict = {"source": "df", "rows": rows, "columns": "parameters"}
            ds = data_sources.DataSource(
                init_config, "ds_name", source_dict, data_source_dfs={"df": df}
            )
            return ds

        return _source_obj

    def test_node_dict_from_one_param(self, source_obj):
        df_dict = {"available_area": {("foo1", "bar1"): 1, ("foo2", "bar2"): 2}}
        tech_dict = calliope.AttrDict({"bar1": {}, "bar2": {}})
        node_dict = source_obj(df_dict).node_dict(tech_dict)

        assert node_dict == {
            "foo1": {"techs": {"bar1": None}},
            "foo2": {"techs": {"bar2": None}},
        }

    def test_node_dict_from_two_param(self, source_obj):
        df_dict = {
            "available_area": {("foo1", "bar1"): 1, ("foo1", "bar2"): 2},
            "other_param": {("foo2", "bar2"): 1},
        }
        tech_dict = calliope.AttrDict({"bar1": {}, "bar2": {}})
        node_dict = source_obj(df_dict).node_dict(tech_dict)

        assert node_dict == {
            "foo1": {"techs": {"bar1": None, "bar2": None}},
            "foo2": {"techs": {"bar2": None}},
        }

    def test_node_dict_extra_dim_in_param(self, source_obj):
        df_dict = {
            "available_area": {("foo1", "bar1", "baz1"): 1, ("foo2", "bar2", "baz2"): 2}
        }
        tech_dict = calliope.AttrDict({"bar1": {}, "bar2": {}})
        node_dict = source_obj(df_dict, rows=["nodes", "techs", "carriers"]).node_dict(
            tech_dict
        )

        assert node_dict == {
            "foo1": {"techs": {"bar1": None}},
            "foo2": {"techs": {"bar2": None}},
        }

    def test_node_dict_node_not_in_ds(self, source_obj):
        node_tech_df_dict = {"my_param": {("foo1", "bar1"): 1, ("foo1", "bar2"): 2}}
        node_df_dict = {"available_area": {"foo2": 1}}
        tech_dict = calliope.AttrDict({"bar1": {}, "bar2": {}})
        node_tech_ds = source_obj(node_tech_df_dict)
        node_ds = source_obj(node_df_dict, rows="nodes")
        node_tech_ds.dataset = node_tech_ds.dataset.merge(node_ds.dataset)

        node_dict = node_tech_ds.node_dict(tech_dict)
        assert node_dict == {
            "foo1": {"techs": {"bar1": None, "bar2": None}},
            "foo2": {"techs": {}},
        }

    def test_node_dict_no_info(self, source_obj):
        df_dict = {"param": {"foo1": 1, "foo2": 2}}
        tech_dict = calliope.AttrDict(
            {"bar1": {"base_tech": "transmission"}, "bar2": {}}
        )
        node_dict = source_obj(df_dict, rows="techs").node_dict(tech_dict)

        assert node_dict == {}

    def test_transmission_tech_with_nodes(self, source_obj):
        df_dict = {"param": {("foo1", "bar1"): 1, ("foo2", "bar2"): 2}}
        tech_dict = calliope.AttrDict(
            {"bar1": {"base_tech": "transmission"}, "bar2": {}}
        )

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            source_obj(df_dict).node_dict(tech_dict)

        check_error_or_warning(
            excinfo,
            "Cannot define transmission technology data over the `nodes` dimension",
        )
