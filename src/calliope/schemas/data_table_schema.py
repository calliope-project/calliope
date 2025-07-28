# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for data table definition."""

from pydantic import Field, model_validator
from typing_extensions import Self

from calliope.schemas.general import (
    AttrStr,
    CalliopeBaseModel,
    CalliopeDictModel,
    UniqueList,
)
from calliope.util.tools import listify


class CalliopeDataTable(CalliopeBaseModel):
    """Data table schema."""

    model_config = {"title": "Data table schema"}

    data: str
    """
    Absolute or relative filepath.
    Relative paths are based on the model config file used to initialise the model.
    """
    rows: None | AttrStr | UniqueList[AttrStr] = None
    """
    Names of dimensions defined row-wise.
    Each name should correspond to a column in your data that contains index items.
    These columns must be to the left of the columns containing your data.
    """
    columns: None | AttrStr | UniqueList[AttrStr] = None
    """
    Names of dimensions defined column-wise.
    Each name should correspond to a row in your data that contains index items.
    These rows must be above the rows containing your data.
    """
    select: None | dict[AttrStr, AttrStr | UniqueList[AttrStr]] = None
    """
    Select one or more index item from a dimension.
    Selection takes place before `drop` and `add_dims`, so you can select a single
    value from a data dimension and then drop the dimension so it doesn't find its way
    through to the final dataset.
    """
    drop: None | AttrStr | UniqueList[AttrStr] = None
    """
    Enables removing rows and/or columns that contain irrelevant data/metadata.
    These could include comments on the source of the data, the data license, or the parameter units.
    You can also drop a dimension and then reintroduce it in `add_dims`, but with different index items.
    """
    add_dims: None | dict[AttrStr, AttrStr | list[AttrStr]] = None
    """
    Data dimensions to add after loading in the array.
    These allow you to use the same file to assign values to different parameters/dimension index items
    (e.g., setting `flow_cap_min` and `flow_cap_max` to the same value),
    or to add a dimension which would otherwise be a column containing the same information in each row
    (e.g., assigning the cost class to monetary for a file containing cost data).
    """
    rename_dims: None | dict[AttrStr, AttrStr] = None
    """
    Mapping between dimension names in the data table being loaded to equivalent Calliope dimension names.
    For instance, the "time" column in the data table would need to be mapped to "timesteps": `{"time": "timesteps"}`.
    """

    @model_validator(mode="after")
    def check_row_and_columns(self) -> Self:
        """Ensure users specify a valid data table shape."""
        drop = set(listify(self.drop))
        rows = set(listify(self.rows)) - drop
        columns = set(listify(self.columns)) - drop
        if not rows and not columns:
            raise ValueError("Either row or columns must be defined (and not dropped).")
        elif rows & columns:
            raise ValueError("Rows and columns must not overlap.")

        if self.add_dims:
            if self.add_dims.keys() & (rows | columns):
                raise ValueError("Added dimensions must not be in columns or rows.")

        if self.rename_dims:
            if set(self.rename_dims.values()) - (rows | columns):
                raise ValueError(
                    "Renamed dimensions must be in either rows or columns."
                )
        return self


class CalliopeDataTables(CalliopeDictModel):
    """Calliope input data table dictionary."""

    root: dict[AttrStr, CalliopeDataTable] = Field(default_factory=dict)
