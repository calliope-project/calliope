# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Preprocessing functionality."""

import logging
from collections.abc import Hashable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import NotRequired, TypedDict

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.io import load_config
from calliope.util.tools import listify, relative_path

LOGGER = logging.getLogger(__name__)


AXIS_T = Literal["columns", "index"]


class DataTableDict(TypedDict):
    """Uniform dictionary for data tables."""

    rows: NotRequired[str | list[str]]
    columns: NotRequired[str | list[str]]
    data: str
    df: NotRequired[str]
    rename_dims: NotRequired[dict[str, str]]
    add_dims: NotRequired[dict[str, str | list[str]]]
    select: NotRequired[dict[str, str | bool | int]]
    drop: NotRequired[Hashable | list[Hashable]]


class DataTable:
    """Class for in memory data handling."""

    MESSAGE_TEMPLATE = "(data_tables, {name}) | {message}."
    PARAMS_TO_INITIALISE_YAML = ["base_tech", "link_to", "link_from"]

    def __init__(
        self,
        table_name: str,
        data_table: DataTableDict,
        data_table_dfs: dict[str, pd.DataFrame] | None = None,
        model_definition_path: str | Path | None = None,
    ):
        """Load and format a data table from file / in-memory object.

        Args:
            table_name (str): name of the data table.
            data_table (DataTableDict): validated data table definition dictionary.
            data_table_dfs (dict[str, pd.DataFrame] | None, optional):
                If given, a dictionary mapping table names in `data_table` to in-memory pandas DataFrames.
                Defaults to None.
            model_definition_path (Path, optional):
                If given, the path to the model definition YAML file, relative to which data table filepaths will be set.
                If None, relative data table filepaths will be considered relative to the current working directory.
                Defaults to None.
        """
        self.input = data_table
        self.dfs = data_table_dfs if data_table_dfs is not None else dict()
        self.model_definition_path = model_definition_path
        self.columns = self._listify_if_defined("columns")
        self.index = self._listify_if_defined("rows")
        self._name = table_name
        self.protected_params = load_config("protected_parameters.yaml")

        if ".csv" in Path(self.input["data"]).suffixes:
            df = self._read_csv()
        else:
            df = self.dfs[self.input["data"]]

        self.dataset = self._df_to_ds(df)

    @property
    def name(self):
        """Data table name."""
        return self._name

    def drop(self, name: str):
        """Drop a data in-place from the data table.

        Args:
            name (str): Name of data array to drop.
        """
        self.dataset = self.dataset.drop_vars(name, errors="ignore")

    def tech_dict(self) -> tuple[AttrDict, AttrDict]:
        """Create a dummy technology definition dictionary from the dataset's "techs" dimension.

        This definition dictionary will ensure that the minimal YAML content is still possible.

        This function should be accessed _before_ `self.node_dict`.

        """
        tech_dict = AttrDict(
            {k: {} for k in self.dataset.get("techs", xr.DataArray([])).values}
        )
        base_tech_data = AttrDict()
        for param in self.PARAMS_TO_INITIALISE_YAML:
            if param in self.dataset:
                base_tech_dict = self.dataset[param].to_dataframe().dropna().T.to_dict()
                base_tech_data.union(base_tech_dict)

        return tech_dict, base_tech_data

    def node_dict(self, techs_incl_inheritance: AttrDict) -> AttrDict:
        """Create a dummy node definition dictionary from the dimensions defined across all data tables.

        This definition dictionary will ensure that the minimal YAML content is still possible.

        This function should be run _after_ `self._update_tech_def_from_data_table`.

        Args:
            techs_incl_inheritance (AttrDict):
                Technology definition dictionary which is a union of any YAML definition and the result of calling `self.tech_dict` across all data tables.
                Technologies should have their definition inheritance resolved.
        """
        node_tech_vars = self.dataset[
            [
                k
                for k, v in self.dataset.data_vars.items()
                if "nodes" in v.dims and "techs" in v.dims
            ]
        ]
        if not node_tech_vars:
            return AttrDict()

        other_dims = [i for i in node_tech_vars.dims if i not in ["nodes", "techs"]]

        is_defined = node_tech_vars.notnull().any(other_dims).to_dataframe().any(axis=1)

        node_tech_dict = AttrDict({i: {"techs": {}} for i in self.dataset.nodes.values})
        for node, tech_dict in node_tech_dict.items():
            try:
                techs_this_node = is_defined[is_defined].xs(node, level="nodes").index
            except KeyError:
                continue
            for tech in techs_this_node:
                if (
                    techs_incl_inheritance[tech].get("base_tech", None)
                    == "transmission"
                ):
                    self._raise_error(
                        "Cannot define transmission technology data over the `nodes` dimension"
                    )
                else:
                    tech_dict["techs"][tech] = None

        return node_tech_dict

    def lookup_dict_from_param(self, param: str, lookup_dim: str) -> AttrDict:
        """Convert "lookup" data loaded from file into YAML-esque format.

        Lookup parameters are those where in YAML the dimension index is the parameter value.

        For example,

        ```yaml
        techs:
          bar:
            carrier_in: baz
        ```

        is translated to the internal data model as:

        ```
        techs | carriers | carrier_in
        bar   | baz      | 1
        ```

        This method converts the tabular data back to the dictionary format.

        !!! note
            Only `techs` and `lookup_dim` will be retained in the resulting dictionary.
            Other dimensions will be dropped.

        Args:
            param (str):
                Parameter in `self.dataset` to search for.
                If found, create a dictionary representation.
            lookup_dim (str):
                The dimension to pivot the data table on.
                The values in this dimension will become the values in the dictionary.

        Returns:
            AttrDict:
                If present in `self.dataset`, a valid Calliope YAML format for the parameter.
                If not, an empty dictionary.
        """

        def __extract_data(grouped_series):
            index_items = sorted(
                list(set(grouped_series.dropna(subset=[param])[lookup_dim].values))
            )
            if len(index_items) == 1:
                index_items = index_items[0]
            if index_items:
                return {param: index_items}
            else:
                return np.nan

        lookup_dict = AttrDict()
        if param not in self.dataset:
            return lookup_dict
        elif (
            "techs" not in self.dataset[param].dims
            or lookup_dim not in self.dataset[param].dims
        ):
            self._raise_error(
                f"Loading {param} with missing dimension(s). "
                f"Must contain `techs` and `{lookup_dim}`, received: {self.dataset[param].dims}"
            )
        else:
            lookup_dict.union(
                self.dataset[param]
                .to_series()
                .reset_index(lookup_dim)
                .groupby("techs")
                .apply(__extract_data)
                .dropna()
                .to_dict()
            )

        return lookup_dict

    def _read_csv(self) -> pd.DataFrame:
        """Load data from CSV.

        Returns:
            pd.DataFrame: Loaded data without any processing.
        """
        filename = self.input["data"]

        if self.columns is None:
            self._log(
                "No column dimensions have been defined. "
                "We will still assume the first line of the CSV file to contain a header row with index dimensions",
                level="debug",
            )
        header = [0] if self.columns is None else list(range(len(self.columns)))
        index_col = None if self.index is None else list(range(len(self.index)))

        filepath = relative_path(self.model_definition_path, filename)
        df = pd.read_csv(filepath, encoding="utf-8", header=header, index_col=index_col)
        return df

    def _df_to_ds(self, df: pd.DataFrame) -> xr.Dataset:
        """Process loaded pandas dataframes into tidy dataframes and then into xarray datasets.

        Args:
            df (pd.DataFrame): Result of `self._read_csv` or an in-memory object.

        Returns:
            xr.Dataset:
                `parameters` dimension in `df` are data variables and all other dimensions are data coordinates.
        """
        if not isinstance(df, pd.DataFrame):
            self._raise_error(
                "Data table must be a pandas DataFrame. "
                "If you are providing an in-memory object, ensure it is not a pandas Series by calling the method `to_frame()`"
            )

        tdf: pd.Series
        axis_names: dict[AXIS_T, None | list[str]] = {
            "columns": self.columns,
            "index": self.index,
        }
        squeeze_me: dict[AXIS_T, bool] = {
            "columns": self.columns is None,
            "index": self.index is None,
        }
        for axis, names in axis_names.items():
            if names is None and len(getattr(df, axis).names) != 1:
                self._raise_error(f"Expected a single {axis} level in loaded data.")
            elif names is not None:
                df = self._rename_axes(df, axis, names)

        for axis, squeeze in squeeze_me.items():
            if squeeze:
                df = df.squeeze(axis=axis)

        if isinstance(df, pd.DataFrame):
            tdf = df.stack(tuple(df.columns.names), future_stack=True).dropna()
        else:
            tdf = df

        if self.input.get("select", None) is not None:
            selector = tuple(
                (
                    listify(self.input["select"][name])
                    if name in self.input["select"]
                    else slice(None)
                )
                for name in tdf.index.names
            )
            tdf = tdf.loc[selector]

        if self.input.get("drop", None) is not None:
            tdf = tdf.droplevel(self.input["drop"])

        if self.input.get("add_dims", None) is not None:
            for dim_name, index_items in self.input["add_dims"].items():
                index_items = listify(index_items)
                tdf = pd.concat(
                    [tdf for _ in index_items], keys=index_items, names=[dim_name]
                )
        self._check_processed_tdf(tdf)
        self._check_for_protected_params(tdf)

        if tdf.index.names == ["parameters"]:  # unindexed parameters
            ds = xr.Dataset(tdf.to_dict())
        else:
            ds = tdf.unstack("parameters").infer_objects().to_xarray()

        self._log(f"Loaded arrays:\n{ds}")
        return ds

    def _rename_axes(
        self, df: pd.DataFrame, axis: AXIS_T, names: list[str]
    ) -> pd.DataFrame:
        """Check and rename DataFrame index and column names according to data table definition.

        Args:
            df (pd.DataFrame): Loaded data table as a DataFrame.
            axis (Literal[columns, index]): DataFrame axis.
            names (list[str] | None): Expected dimension names along `axis`.

        Returns:
            pd.DataFrame: `df` with all dimensions on `axis` appropriately named.
        """
        if len(getattr(df, axis).names) != len(names):
            self._raise_error(f"Expected {len(names)} {axis} levels in loaded data.")
        mapper = self.input.get("rename_dims", {})
        if mapper:
            df.rename_axis(inplace=True, **{axis: mapper})
        self._compare_axis_names(getattr(df, axis).names, names, axis)
        df.rename_axis(inplace=True, **{axis: names})

        return df

    def _check_for_protected_params(self, tdf: pd.Series):
        """Raise an error if any defined parameters are in a pre-configured set of _protected_ parameters.

        See keys of `self.protected_params` for list of protected parameters.

        Args:
            tdf (pd.Series): Tidy dataframe in which to search for parameters (within the `parameters` index level).
        """
        invalid_params = tdf.index.get_level_values("parameters").intersection(
            list(self.protected_params.keys())
        )
        if not invalid_params.empty:
            extra_info = set(self.protected_params[k] for k in invalid_params)
            exceptions.print_warnings_and_raise_errors(
                errors=list(extra_info), during=f"data table loading ({self.name})"
            )

    def _check_processed_tdf(self, tdf: pd.Series):
        if "parameters" not in tdf.index.names:
            self._raise_error(
                "The `parameters` dimension must exist in on of `rows`, `columns`, or `add_dims`."
            )

        duplicated_index = tdf.index.duplicated()
        if any(duplicated_index):
            self._raise_error(
                f"Duplicate index items found: {tdf.index[duplicated_index]}"
            )

        unique_index_names = set(tdf.index.names)
        if len(unique_index_names) != len(tdf.index.names):
            self._raise_error(f"Duplicate dimension names found: {tdf.index.names}")

    def _raise_error(self, message):
        """Format error message and then raise Calliope ModelError."""
        raise exceptions.ModelError(
            self.MESSAGE_TEMPLATE.format(name=self.name, message=message)
        )

    def _log(self, message, level="debug"):
        """Format log message and then log to `level`."""
        getattr(LOGGER, level)(
            self.MESSAGE_TEMPLATE.format(name=self.name, message=message)
        )

    def _listify_if_defined(self, key: str) -> list | None:
        """If `key` is in data sourtablece definition dictionary, return values as a list.

        If values are not yet an iterable, they will be coerced to an iterable of length 1.
        If they are an iterable, they will be coerced to a list.

        Args:
            key (str): Definition dictionary key whose value are to be returned as a list.
            default (Literal[None, 0]): Either zero or None

        Returns:
            list | None: If `key` not defined in data table, return None, else return values as a list.
        """
        vals = self.input.get(key, None)
        if vals is not None:
            vals = listify(vals)
        return vals

    def _compare_axis_names(self, loaded_names: list, defined_names: list, axis: str):
        """Check loaded axis level names compared to those given by `rows` and `columns` in data table definition dictionary.

        The data file / in-memory object does not need to have any level names defined,
        but if they _are_ defined then they must match those given in the data table definition dictionary.

        Args:
            loaded_names (list): Names as defined in the loaded data file / in-memory object.
            defined_names (list): Names as defined in the data table dictionary.
            axis (str): Axis on which the names are levels.
        """
        if any(
            name is not None
            and not isinstance(name, int)
            and name != defined_names[idx]
            for idx, name in enumerate(loaded_names)
        ):
            self._raise_error(
                f"Trying to set names for {axis} but names in the file do no match names provided |"
                f" in file: {loaded_names} | defined: {defined_names}"
            )
