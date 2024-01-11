# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

import logging
from pathlib import Path
from typing import Hashable, Optional

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import NotRequired, TypedDict

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.core.io import load_config
from calliope.preprocess import time
from calliope.util.tools import listify, relative_path

LOGGER = logging.getLogger(__name__)


class DataSourceDict(TypedDict):
    rows: NotRequired[str | list[str]]
    columns: NotRequired[str | list[str]]
    source: str
    df: NotRequired[str]
    add_dimensions: NotRequired[dict[str, str | list[str]]]
    drop: Hashable | list[Hashable]
    sel_drop: dict[str, str | bool | int]


class DataSource:
    def __init__(
        self,
        model_config: dict,
        data_source: DataSourceDict,
        data_source_dfs: Optional[dict[str, pd.DataFrame]],
        model_definition_path: Optional[Path] = None,
    ):
        """Load and format a data source from file / in-memory object.

        Args:
            model_config (dict): Model initialisation configuration dictionary.
            data_source (DataSourceDict): Data source definition dictionary.
            data_source_dfs (Optional[dict[str, pd.DataFrame]]):
                If given, a dictionary mapping source names in `data_source` to in-memory pandas DataFrames.
            model_definition_path (Optional[Path], optional):
                If given, the path to the model definition YAML file, relative to which data source filepaths will be set.
                If None, relative data source filepaths will be considered relative to the current working directory.
                Defaults to None.
        """
        self.input = data_source
        self.dfs = data_source_dfs if data_source_dfs is not None else dict()
        self.model_definition_path = model_definition_path
        self.config = model_config

        self.columns = self._listify_if_defined("columns")
        self.index = self._listify_if_defined("rows")
        self._name = self.input["source"]
        self.protected_params = load_config("protected_parameters.yaml")

        if ".csv" in Path(self.name).suffixes:
            df = self._read_csv()
        else:
            df = self.dfs[self.name]

        self.dataset = self._df_to_ds(df)

    @property
    def name(self):
        "Data source name"
        return self.input["source"]

    def tech_dict(self) -> tuple[AttrDict, AttrDict]:
        """Create a dummy technology definition dictionary from the dataset's "techs" dimension.

        This definition dictionary will ensure that the minimal YAML content is still possible.

        This function should be accessed _before_ `self.node_dict`.

        """

        tech_dict = AttrDict(
            {k: {} for k in self.dataset.get("techs", xr.DataArray([])).values}
        )
        base_tech_data = AttrDict()
        if "parent" in self.dataset:
            parent_dict = AttrDict(self.dataset["parent"].to_dataframe().T.to_dict())
            base_tech_data.union(parent_dict)
        carrier_info_dict = self._carrier_info_dict_from_model_data_var(self.dataset)
        base_tech_data.union(carrier_info_dict)

        return tech_dict, base_tech_data

    def node_dict(
        self, techs_incl_inheritance: AttrDict, base_tech_data: AttrDict
    ) -> AttrDict:
        """Create a dummy node definition dictionary from the dimensions defined across all data sources.

        In addition, if transmission technologies are defined then update `to`/`from` params in the technology definition dictionary.

        This definition dictionary will ensure that the minimal YAML content is still possible.

        This function should be run _after_ `self._update_tech_def_from_data_source`.

        Args:
            techs_incl_inheritance (AttrDict):
                Technology definition dictionary which is a union of any YAML definition and the result of calling `self.tech_dict` across all data sources.
                Technologies should have their entire definition inheritance chain resolved.
            base_tech_data (AttrDict):
                Technology definition dictionary containing only a subset of parameters _if_ they have been defined in data sources.
                These parameters include `parent`, `carrier_in`, `carrier_out`.
                After calling this function, and if any transmission technologies are defined in this data source,
                this dictionary will also contain `to` and `from` parameters.

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

        is_defined = (
            node_tech_vars.fillna(False).any(other_dims).to_dataframe().any(axis=1)
        )
        node_tech_dict = AttrDict({i: {"techs": {}} for i in self.dataset.nodes.values})
        for node, tech_dict in node_tech_dict.items():
            try:
                techs_this_node = is_defined[is_defined].xs(node, level="nodes").index
            except KeyError:
                continue
            for tech in techs_this_node:
                if techs_incl_inheritance[tech].get("parent", None) == "transmission":
                    if "from" in base_tech_data[tech]:
                        base_tech_data.set_key(f"{tech}.to", node)
                    else:
                        base_tech_data.set_key(f"{tech}.from", node)
                else:
                    tech_dict["techs"][tech] = None

        return node_tech_dict

    def _read_csv(self) -> pd.DataFrame:
        """Load data from CSV.

        Returns:
            pd.DataFrame: Loaded data without any processing.
        """
        filename = self.input["source"]
        csv_reader_kwargs = {"header": self.columns, "index_col": self.index}
        for axis, names in csv_reader_kwargs.items():
            if names is not None:
                csv_reader_kwargs[axis] = [i for i, _ in enumerate(names)]

        filepath = relative_path(self.model_definition_path, filename)
        df = pd.read_csv(filepath, encoding="utf-8", **csv_reader_kwargs)
        return df

    def _df_to_ds(self, df: pd.DataFrame) -> xr.Dataset:
        """Process loaded pandas dataframes into tidy dataframes and then into xarray datasets.

        Args:
            df (pd.DataFrame): Result of `self._read_csv` or an in-memory object.

        Returns:
            xr.Dataset:
                `parameters` dimension in `df` are data variables and all other dimensions are data coordinates.
        """
        for axis, names in {"columns": self.columns, "index": self.index}.items():
            if names is None:
                df = df.squeeze(axis=axis)
            else:
                self._compare_axis_names(getattr(df, axis).names, names, axis)
                df.rename_axis(inplace=True, **{axis: names})

        tdf: pd.Series
        if isinstance(df, pd.DataFrame):
            tdf = df.stack(df.columns.names)  # type: ignore
        else:
            tdf = df

        if "drop" in self.input.keys():
            tdf = tdf.droplevel(self.input["drop"])

        if "sel_drop" in self.input.keys():
            tdf = tdf.xs(
                tuple(self.input["sel_drop"].values()),
                level=tuple(self.input["sel_drop"].keys()),
                drop_level=True,
            )

        if "add_dimensions" in self.input.keys():
            for dim_name, index_items in self.input["add_dimensions"].items():
                index_items = listify(index_items)
                tdf = pd.concat(
                    [tdf for _ in index_items], keys=index_items, names=[dim_name]
                )

        if "parameters" not in tdf.index.names:
            self._raise_error(
                "The `parameters` dimension must exist in on of `rows`, `columns`, or `add_dimensions`."
            )
        self._check_for_protected_params(tdf)

        if tdf.index.names == ["parameters"]:  # unindexed parameters
            ds = xr.Dataset(tdf.to_dict())
        else:
            ds = tdf.unstack("parameters").to_xarray()
        ds = time.clean_data_source_timeseries(ds, self.config, self.name)

        self._log("Loaded arrays:\n{ds.data_vars}")
        return ds

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
                errors=list(extra_info), during=f"data source loading ({self.name})"
            )

    def _raise_error(self, message):
        "Format error message and then raise Calliope ModelError."
        raise exceptions.ModelError(f"(data_sources, {self.name}) | {message}.")

    def _log(self, message, level="debug"):
        "Format log message and then log to `level`."
        getattr(LOGGER, level)(f"(data_sources, {self.name}) | {message}")

    def _listify_if_defined(self, key: str) -> Optional[list]:
        """If `key` is in data source definition dictionary, return values as a list.

        If values are not yet an iterable, they will be coerced to an iterable of length 1.
        If they are an iterable, they will be coerced to a list.

        Args:
            key (str): Definition dictionary key whose value are to be returned as a list.

        Returns:
            Optional[list]: If `key` not defined in data source, return None, else return values as a list.
        """
        vals = self.input.get(key, None)
        if vals is not None:
            vals = listify(vals)
        return vals

    def _compare_axis_names(self, loaded_names: list, defined_names: list, axis: str):
        """Check loaded axis level names compared to those given by `rows` and `columns` in data source definition dictionary.

        The data file / in-memory object does not need to have any level names defined,
        but if they _are_ defined then they must match those given in the data source definition dictionary.

        Args:
            loaded_names (list): Names as defined in the loaded data file / in-memory object.
            defined_names (list): Names as defined in the data source dictionary.
            axis (str): Axis on which the names are levels.
        """
        if any(
            name is not None
            and not isinstance(name, int)
            and name != defined_names[idx]
            for name, idx in enumerate(loaded_names)
        ):
            self._raise_error(
                f"Trying to set names for {axis} but names in the file do no match names provided | in file: {loaded_names} | defined: {defined_names}"
            )

    @staticmethod
    def _carrier_info_dict_from_model_data_var(data_source: xr.Dataset) -> AttrDict:
        """Convert `carrier_in`/`carrier_out` loaded from a data source into YAML-esque format.

        Only used when either variable is defined in a data source.
        Conversion from e.g.,

        ```
        nodes | techs | carriers | carrier_in
        foo   | bar   | baz      | 1
        ```

        to
        ```yaml
        techs:
          bar:
            carrier_in: baz
        ```

        !!! note
            This will drop the `nodes` dimension entirely as we expect these parameters to be defined at the tech level in YAML.

        Args:
            data_source (xr.Dataset): data loaded from file that may contain `carrier_in`/`carrier_out`.

        Returns:
            AttrDict:
                If present in `data_source`, a valid Calliope YAML format for `carrier_in`/`carrier_out`.
                If not, an empty dictionary.
        """

        def __extract_carriers(grouped_series):
            carriers = list(
                set(
                    grouped_series.dropna(
                        subset=[f"carrier_{direction}"]
                    ).carriers.values
                )
            )
            if len(carriers) == 1:
                carriers = carriers[0]
            if carriers:
                return {f"carrier_{direction}": carriers}
            else:
                return np.nan

        carrier_info_dict = AttrDict()
        for direction in ["in", "out"]:
            if f"carrier_{direction}" not in data_source:
                continue
            carrier_info_dict.union(
                AttrDict(
                    data_source[f"carrier_{direction}"]
                    .to_series()
                    .reset_index("carriers")
                    .groupby("techs")
                    .apply(__extract_carriers)
                    .dropna()
                    .to_dict()
                )
            )

        return carrier_info_dict