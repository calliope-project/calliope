"""Test data table schema validation."""

import pytest
from pydantic import ValidationError

from calliope.attrdict import AttrDict
from calliope.schemas.data_table_schema import DataTable

from .common.util import check_error_or_warning

FULL_TABLE_CONFIG = """
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
template: some_template
"""


@pytest.mark.parametrize(
    "data_table",
    [{"rows": "timesteps"}, {"rows": "timesteps", "columns": ["techs", "nodes"]}],
)
def test_path_not_provided(data_table):
    """Not providing the path should result in a failure."""
    with pytest.raises(ValidationError):
        DataTable(**data_table)


@pytest.mark.parametrize("data_table", [{"data": "foo"}])
def test_incomplete_column_or_row(data_table):
    """Not providing either rows or columns is invalid."""
    with pytest.raises(ValidationError) as excinfo:
        DataTable(**data_table)
    assert check_error_or_warning(
        excinfo, "Either row or columns must be defined for data_table."
    )


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        ("nodes", "nodes"),
        (["nodes", "techs"], "techs"),
        (["nodes", "techs", "params"], ["params", "costs"]),
    ],
)
def test_row_column_overlap(rows, columns):
    """Rows and columns must not share any similar values."""
    with pytest.raises(ValidationError) as excinfo:
        DataTable(data="foobar", rows=rows, columns=columns)
    assert check_error_or_warning(excinfo, "Rows and columns must not overlap.")


@pytest.mark.parametrize(
    ("rows", "columns", "add_dims"), [("nodes", None, {"nodes": "MEX"})]
)
def test_add_dims_overlap(rows, columns, add_dims):
    with pytest.raises(ValidationError) as excinfo:
        DataTable(data="foo", rows=rows, columns=columns, add_dims=add_dims)
    assert check_error_or_warning(
        excinfo, "Added dimensions must not be in columns or rows."
    )


@pytest.mark.parametrize("data_table", [FULL_TABLE_CONFIG])
def test_full_table_config(data_table):
    """Test a fully fledged data table configuration."""
    DataTable(**AttrDict.from_yaml_string(data_table))
