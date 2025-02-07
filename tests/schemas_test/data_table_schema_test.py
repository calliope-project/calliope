"""Test data table schema validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from calliope import io
from calliope.attrdict import AttrDict
from calliope.schemas.data_table_schema import CalliopeDataTable

from ..common.util import check_error_or_warning


@pytest.fixture
def full_data_table_config():
    return """data: time_varying_df
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


@pytest.fixture
def model_yaml_data_tables() -> AttrDict:
    return io.read_rich_yaml(
        Path(__file__).parent.parent
        / "common"
        / "national_scale_from_data_tables"
        / "model.yaml"
    )


@pytest.mark.parametrize(
    "data_table",
    [{"rows": "timesteps"}, {"rows": "timesteps", "columns": ["techs", "nodes"]}],
)
def test_path_not_provided(data_table):
    """Not providing the path should result in a failure."""
    with pytest.raises(ValidationError):
        CalliopeDataTable(**data_table)


@pytest.mark.parametrize("data_table", [{"data": "foo"}])
def test_incomplete_column_or_row(data_table):
    """Not providing either rows or columns is invalid."""
    with pytest.raises(ValidationError) as excinfo:
        CalliopeDataTable(**data_table)
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
        CalliopeDataTable(data="foobar", rows=rows, columns=columns)
    assert check_error_or_warning(excinfo, "Rows and columns must not overlap.")


@pytest.mark.parametrize(
    ("rows", "columns", "add_dims"), [("nodes", None, {"nodes": "MEX"})]
)
def test_add_dims_overlap(rows, columns, add_dims):
    with pytest.raises(ValidationError) as excinfo:
        CalliopeDataTable(data="foo", rows=rows, columns=columns, add_dims=add_dims)
    assert check_error_or_warning(
        excinfo, "Added dimensions must not be in columns or rows."
    )


def test_full_table_config(full_data_table_config):
    """Test a fully fledged data table configuration."""
    CalliopeDataTable(**io.read_rich_yaml(full_data_table_config))


def test_data_table_model(model_yaml_data_tables):
    """Data table validation must conform to expected usage."""
    for data_table in model_yaml_data_tables["data_tables"].values():
        CalliopeDataTable(**data_table)
