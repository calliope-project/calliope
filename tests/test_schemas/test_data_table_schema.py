"""Test data table schema validation."""

import pytest
from pydantic import ValidationError

from calliope.io import read_rich_yaml
from calliope.schemas.data_table_schema import CalliopeDataTable

from ..common.util import check_error_or_warning


class TestCalliopeDataTable:
    @pytest.fixture
    def full_data_table_config(self):
        return read_rich_yaml("""
    data: time_varying_df
    rows: timesteps
    columns: [comment, nodes, techs]
    select:
        nodes: [node1, node2]
        techs: pv
    drop: comment
    add_dims:
        parameters: something
        costs: monetary
    rename_dims:
        location: nodes
    """)

    @pytest.mark.parametrize(
        "data_table",
        [{"rows": "timesteps"}, {"rows": "timesteps", "columns": ["techs", "nodes"]}],
    )
    def test_path_not_provided(self, data_table):
        """Not providing the path should result in a failure."""
        with pytest.raises(ValidationError, match="validation error for Data table"):
            CalliopeDataTable(**data_table)

    @pytest.mark.parametrize("data_table", [{"data": "foo"}])
    def test_incomplete_column_or_row(self, data_table):
        """Not providing either rows or columns is invalid."""
        with pytest.raises(ValidationError) as excinfo:
            CalliopeDataTable(**data_table)
        assert check_error_or_warning(
            excinfo, "Either row or columns must be defined (and not dropped)."
        )

    @pytest.mark.parametrize(
        ("rows", "columns"),
        [
            ("nodes", "nodes"),
            (["nodes", "techs"], "techs"),
            (["nodes", "techs", "params"], ["params", "costs"]),
        ],
    )
    def test_row_column_overlap(self, rows, columns):
        """Rows and columns must not share any similar values."""
        with pytest.raises(ValidationError) as excinfo:
            CalliopeDataTable(data="foobar", rows=rows, columns=columns)
        assert check_error_or_warning(excinfo, "Rows and columns must not overlap.")

    @pytest.mark.parametrize(
        ("rows", "columns", "add_dims"), [("nodes", None, {"nodes": "MEX"})]
    )
    def test_add_dims_overlap(self, rows, columns, add_dims):
        with pytest.raises(ValidationError) as excinfo:
            CalliopeDataTable(data="foo", rows=rows, columns=columns, add_dims=add_dims)
        assert check_error_or_warning(
            excinfo, "Added dimensions must not be in columns or rows."
        )

    def test_full_table_config(self, full_data_table_config):
        """Test a fully fledged data table configuration."""
        CalliopeDataTable(**full_data_table_config)

    def test_example_models(self, model_def):
        """Test the schema against example and test model definitions."""
        if "data_tables" in model_def:
            for data_table_def in model_def["data_tables"].values():
                CalliopeDataTable(**data_table_def)

    @pytest.mark.parametrize(
        ("rename_dims", "drop"),
        [({"bar": "foo"}, None), ({"locations": "nodes"}, "nodes")],
    )
    def test_invalid_rename(self, rename_dims, drop, full_data_table_config):
        """Renaming inexistent dimensions should be impossible."""
        full_data_table_config.update({"drop": drop, "rename_dims": rename_dims})
        with pytest.raises(
            ValidationError,
            match="Renamed dimensions must be in either rows or columns.",
        ):
            CalliopeDataTable(**full_data_table_config)
