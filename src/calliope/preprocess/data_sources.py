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
from calliope.preprocess import model_data, time
from calliope.util.tools import listify, load_config, relative_path

LOGGER = logging.getLogger(__name__)


class DataSource(TypedDict):
    rows: NotRequired[str | list[str]]
    columns: NotRequired[str | list[str]]
    file: str
    add_dimensions: NotRequired[dict[str, str | list[str]]]
    drop: Hashable | list[Hashable]
    sel_drop: dict[str, str | bool | int]


class ModelDefinition(model_data.ModelDefinition):
    data_sources: NotRequired[list[DataSource]]


class DataSourceFactory(model_data.ModelDataFactory):
    def __init__(
        self,
        model_config: dict,
        model_definition: ModelDefinition,
        model_definition_path: Optional[Path] = None,
    ):
        """Take a Calliope model definition dictionary and convert it into an xarray Dataset, ready for
        constraint generation.

        This includes extracting timeseries data from file and resampling/clustering as necessary.

        Args:
            model_config (dict): Model initialisation configuration (i.e., `config.init`).
            model_definition (ModelDefinition): Definition of model nodes and technologies as a dictionary.
            attributes (dict): Attributes to attach to the model Dataset.
            param_attributes (dict[str, dict]): Attributes to attach to the generated model DataArrays.
        """
        super().__init__(model_config, model_definition, {})

        self.model_data = xr.Dataset()
        self.model_definition: ModelDefinition
        self.model_definition_path = model_definition_path
        self.protected_params = load_config("protected_parameters.yaml")

    def load_data_sources(self) -> None:
        """Load data from the `data_sources` list into the model dataset.

        Each subsequent data source in the list will override prior data.

        Raises:
            exceptions.ModelError: All data sources must have a `parameters` dimension.
            exceptions.ModelError: Certain parameters are only valid if defined in YAML; they cannot be defined in data sources.
        """
        datasets: list[xr.Dataset] = []
        for data_source in self.model_definition.pop("data_sources", []):
            datasets.append(self._load_data_source(data_source))

        self._update_model_definition_from_data_sources(datasets)

        self.model_data = xr.merge(
            datasets, compat="override", combine_attrs="no_conflicts"
        )

    def _load_data_source(self, data_source: DataSource):
        filename = data_source["file"]
        columns = self._listify_if_defined(data_source, "columns")
        index = self._listify_if_defined(data_source, "rows")

        csv_reader_kwargs = {"header": columns, "index_col": index}
        for axis, names in csv_reader_kwargs.items():
            if names is not None:
                csv_reader_kwargs[axis] = [i for i, _ in enumerate(names)]

        filepath = relative_path(self.model_definition_path, filename)
        df = pd.read_csv(filepath, encoding="utf-8", **csv_reader_kwargs)

        for axis, names in {"columns": columns, "index": index}.items():
            if names is None:
                df = df.squeeze(axis=axis)
            else:
                self._compare_axis_names(getattr(df, axis).names, names, axis, filename)
                df.rename_axis(inplace=True, **{axis: names})

        tdf: pd.Series
        if isinstance(df, pd.DataFrame):
            tdf = df.stack(df.columns.names)  # type: ignore
        else:
            tdf = df

        if "drop" in data_source.keys():
            tdf = tdf.droplevel(data_source["drop"])

        if "sel_drop" in data_source.keys():
            tdf = tdf.xs(
                tuple(data_source["sel_drop"].values()),
                level=tuple(data_source["sel_drop"].keys()),
                drop_level=True,
            )

        if "add_dimensions" in data_source.keys():
            for dim_name, index_items in data_source["add_dimensions"].items():
                index_items = listify(index_items)
                tdf = pd.concat(
                    [tdf for _ in index_items], keys=index_items, names=[dim_name]
                )

        if "parameters" not in tdf.index.names:
            raise exceptions.ModelError(
                f"(data_sources, {filename}) | The `parameters` dimension must exist in on of `rows`, `columns`, or `add_dimensions`."
            )
        self._check_for_protected_params(tdf, filename)

        if tdf.index.names == ["parameters"]:  # unindexed parameters
            ds = xr.Dataset(tdf.to_dict())
        else:
            ds = tdf.unstack("parameters").to_xarray()
        ds = time.clean_data_source_timeseries(ds, self.config, filename)

        LOGGER.debug(f"(data_sources, {filename}) | Loaded arrays:\n{ds.data_vars}")
        return ds

    def _check_for_protected_params(self, tdf: pd.Series, filename: str):
        invalid_params = tdf.index.get_level_values("parameters").intersection(
            list(self.protected_params.keys())
        )
        if not invalid_params.empty:
            extra_info = set(self.protected_params[k] for k in invalid_params)
            exceptions.print_warnings_and_raise_errors(
                errors=list(extra_info), during=f"data source loading ({filename})"
            )

    def _update_model_definition_from_data_sources(self, datasets: list[xr.Dataset]):
        # We apply these 1. for each dataset separately and 2. for tech defs entirely before node defs.
        # 1. because we want to avoid the affect of dimension broadcasting that will occur when merging the datasets.
        # 2. because we want to ensure any possible `parent` info is available for techs when we update the node definitions.
        for ds in datasets:
            if "techs" in ds.dims:
                self._update_tech_def_from_data_source(ds)
        for ds in datasets:
            if "techs" in ds.dims and "nodes" in ds.dims:
                self._update_node_def_from_data_source(ds)

    def _update_tech_def_from_data_source(self, data_source: xr.Dataset) -> None:
        """Create a dummy technology definition dictionary from the dimensions defined across all data sources.

        This definition dictionary will ensure that the minimal YAML content is still possible.

        This function should be run _before_ `self._update_node_def_from_data_source`.

        Args:
            data_source (xr.Dataset): Data loaded from one file.
        """
        techs = data_source.techs
        base_tech_data = AttrDict({k: {} for k in techs.values})
        if "parent" in data_source:
            parent_dict = AttrDict(data_source["parent"].to_dataframe().T.to_dict())
            base_tech_data.union(parent_dict)
        carrier_info_dict = self._carrier_info_dict_from_model_data_var(data_source)
        base_tech_data.union(carrier_info_dict)
        self.tech_data_from_sources.union(base_tech_data, allow_override=True)

        tech_dict = AttrDict({k: {} for k in techs.values})
        tech_dict.union(
            self.model_definition.get("techs", AttrDict()), allow_override=True
        )
        self.model_definition["techs"] = tech_dict

    def _update_node_def_from_data_source(self, data_source: xr.Dataset) -> None:
        """Create a dummy node definition dictionary from the dimensions defined across all data sources.

        In addition, if transmission technologies are defined then update `to`/`from` params in the technology definition dictionary.

        This definition dictionary will ensure that the minimal YAML content is still possible.

        This function should be run _after_ `self._update_tech_def_from_data_source`.

        Args:
            data_source (xr.Dataset): Data loaded from one file.

        """
        nodes = data_source.nodes

        node_tech_vars = data_source[
            [
                k
                for k, v in data_source.data_vars.items()
                if "nodes" in v.dims and "techs" in v.dims
            ]
        ]
        other_dims = [i for i in node_tech_vars.dims if i not in ["nodes", "techs"]]

        is_defined = (
            node_tech_vars.fillna(False).any(other_dims).to_dataframe().any(axis=1)
        )
        node_tech_dict = AttrDict({i: {"techs": {}} for i in nodes.values})
        for node, tech_dict in node_tech_dict.items():
            try:
                techs_this_node = is_defined[is_defined].xs(node, level="nodes").index
            except KeyError:
                continue
            for tech in techs_this_node:
                (
                    tech_incl_inheritance,
                    _,
                ) = model_data.ModelDataFactory.climb_inheritance_tree(
                    self, self.model_definition["techs"][tech], "techs", tech
                )
                if tech_incl_inheritance.get("parent", None) == "transmission":
                    if "from" in self.tech_data_from_sources[tech]:
                        self.tech_data_from_sources.set_key(f"{tech}.to", node)
                    else:
                        self.tech_data_from_sources.set_key(f"{tech}.from", node)
                else:
                    tech_dict["techs"][tech] = None

        node_tech_dict.union(
            self.model_definition.get("nodes", AttrDict()), allow_override=True
        )
        self.model_definition["nodes"] = node_tech_dict

    @staticmethod
    def _listify_if_defined(source: DataSource, key: str) -> Optional[list]:
        vals = source.get(key, None)
        if vals is not None:
            vals = listify(vals)
        return vals

    @staticmethod
    def _compare_axis_names(
        loaded_names: list, defined_names: list, axis: str, filename: str
    ):
        if any(
            name is not None
            and not isinstance(name, int)
            and name != defined_names[idx]
            for name, idx in enumerate(loaded_names)
        ):
            raise ValueError(
                f"(data_sources, {filename}) | Trying to set names for {axis} but names in the file do no match names provided | in file: {loaded_names} | defined: {defined_names}"
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
