# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

import itertools
import logging
from copy import deepcopy
from typing import Hashable, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
from geographiclib import geodesic
from typing_extensions import NotRequired, TypedDict

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.preprocess import time
from calliope.util.schema import MODEL_SCHEMA, validate_dict
from calliope.util.tools import listify, relative_path

LOGGER = logging.getLogger(__name__)

DATA_T = Optional[float | int | bool | str] | list[Optional[float | int | bool | str]]

PROTECTED_PARAMS = {
    "to": "`to`/`from` is only valid in YAML definition. Set the `carrier_in` and `carrier_out` parameters of transmission technologies at the nodes they link together to emulate their function when loading from file.",
    "from": "`to`/`from` is only valid in YAML definition. Set the `carrier_in` and `carrier_out` parameters of transmission technologies at the nodes they link together to emulate their function when loading from file.",
    "definition_matrix": "`definition_matrix` is a protected array, it will be generated internally based on the values you assign to the `carrier_in` and `carrier_out` parameters.",
    "inherit": "Cannot use technology/node inheritance (`inherit`) when loading data from file.",
}


class DataSource(TypedDict):
    rows: NotRequired[str | list[str]]
    columns: NotRequired[str | list[str]]
    file: str
    add_dimensions: NotRequired[dict[str, str | list[str]]]
    drop: Hashable | list[Hashable]
    sel_drop: dict[str, str | bool | int]


class Param(TypedDict):
    data: DATA_T
    index: list[list[str]]
    dims: list[str]


class ModelDefinition(TypedDict):
    techs: AttrDict
    nodes: AttrDict
    tech_groups: NotRequired[AttrDict]
    node_groups: NotRequired[AttrDict]
    parameters: NotRequired[AttrDict]
    data_sources: NotRequired[list[DataSource]]


LOGGER = logging.getLogger(__name__)


class ModelDataFactory:
    # TODO: move into yaml syntax and have it be updatable
    LOOKUP_PARAMS = {
        "carrier_in": "carriers",
        "carrier_out": "carriers",
        "carrier_export": "carriers",
    }

    # Output of: sns.color_palette('cubehelix', 10).as_hex()
    _DEFAULT_PALETTE = [
        "#19122b",
        "#17344c",
        "#185b48",
        "#3c7632",
        "#7e7a36",
        "#bc7967",
        "#d486af",
        "#caa9e7",
        "#c2d2f3",
        "#d6f0ef",
    ]

    def __init__(
        self,
        model_config: dict,
        model_definition: ModelDefinition,
        attributes: dict,
        param_attributes: dict[str, dict],
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

        self.config: dict = model_config
        self.model_definition: ModelDefinition = model_definition.copy()
        self.model_data = xr.Dataset(attrs=AttrDict(attributes))
        self._tech_data_source_dict = AttrDict()

        flipped_attributes: dict[str, dict] = dict()
        for key, val in param_attributes.items():
            for subkey, subval in val.items():
                flipped_attributes.setdefault(subkey, {})
                flipped_attributes[subkey][key] = subval
        self.param_attrs = flipped_attributes

    def build(self, timeseries_dfs: Optional[dict[str, pd.DataFrame]]) -> xr.Dataset:
        """Main function used by the calliope Model object to invoke the factory and get it churning out a model dataset.

        Args:
            timeseries_dfs (Optional[dict[str, pd.DataFrame]]): If loading data from pre-loaded dataframes, they need to be provided here.

        Returns:
            xr.Dataset: Built model dataset, including the timeseries dimension.
        """
        self.load_data_sources()
        self.add_node_tech_data()
        self.add_time_dimension(timeseries_dfs)
        self.add_top_level_params()
        self.clean_data_from_undefined_members()
        self.add_colors()
        self.add_link_distances()
        self.resample_time_dimension()
        self.assign_input_attr()
        return self.model_data

    def load_data_sources(self) -> None:
        """Load data from the `data_sources` list into the model dataset.

        Each subsequent data source in the list will override prior data.

        Raises:
            exceptions.ModelError: All data sources must have a `parameters` dimension.
            exceptions.ModelError: Certain parameters are only valid if defined in YAML; they cannot be defined in data sources.
        """
        datasets: list[xr.Dataset] = []
        for data_source in self.model_definition.get("data_sources", []):
            filename = data_source["file"]
            columns = self._listify_if_defined(data_source, "columns")
            index = self._listify_if_defined(data_source, "rows")

            csv_reader_kwargs = {"header": columns, "index_col": index}
            for axis, names in csv_reader_kwargs.items():
                if names is not None:
                    csv_reader_kwargs[axis] = [i for i, _ in enumerate(names)]

            filepath = relative_path(self.config["data_sources_path"], filename)
            df = pd.read_csv(filepath, encoding="latin1", **csv_reader_kwargs)

            for axis, names in {"columns": columns, "index": index}.items():
                if names is None:
                    df = df.squeeze(axis=axis)
                else:
                    self._compare_axis_names(
                        getattr(df, axis).names, names, axis, filename
                    )
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

            invalid_params = tdf.index.get_level_values("parameters").intersection(
                list(PROTECTED_PARAMS.keys())
            )
            if not invalid_params.empty:
                extra_info = set(PROTECTED_PARAMS[k] for k in invalid_params)
                exceptions.print_warnings_and_raise_errors(
                    errors=list(extra_info), during=f"data source loading ({filename})"
                )

            if tdf.index.names == ["parameters"]:  # unindexed parameters
                ds = xr.Dataset(tdf.to_dict())
            else:
                ds = tdf.unstack("parameters").to_xarray()
            ds = time.clean_data_source_timeseries(ds, self.config, filename)
            for da_name, da in ds.data_vars.items():
                da.attrs.update(self.param_attrs.get(da_name, {}))
            LOGGER.debug(f"(data_sources, {filename}) | Loaded arrays:\n{ds.data_vars}")
            datasets.append(ds)

        # We apply these 1. for each dataset separately and 2. for tech defs entirely before node defs.
        # 1. because we want to avoid the affect of dimension broadcasting that will occur when merging the datasets.
        # 2. because we want to ensure any possible `parent` info is available for techs when we update the node definitions.
        for ds in datasets:
            if "techs" in ds.dims:
                self._update_tech_def_from_data_source(ds)
        for ds in datasets:
            if "techs" in ds.dims and "nodes" in ds.dims:
                self._update_node_def_from_data_source(ds)

        if datasets:
            self.model_data = xr.merge(
                [self.model_data, *datasets],
                compat="override",
                combine_attrs="no_conflicts",
            )

    def add_node_tech_data(self):
        """For each node, extract technology definitions and node-level parameters and convert them to arrays.

        The node definition will first be updated according to any defined inheritance (via `inherit`),
        before processing each defined tech (which will also be updated according to its inheritance tree).

        Node and tech definitions will be validated against the model definition schema here.
        """
        active_node_dict = self._inherit_defs("nodes")
        links_at_nodes = self._links_to_node_format(active_node_dict)

        node_tech_data = []
        for node_name, node_data in active_node_dict.items():
            techs_this_node = node_data.pop("techs")
            if techs_this_node is None:
                techs_this_node = AttrDict()
            node_ref_vars = self._get_relevant_node_refs(techs_this_node, node_name)

            techs_this_node_incl_inheritance = self._inherit_defs(
                "techs", techs_this_node, err_message_prefix=f"(nodes, {node_name}), "
            )
            validate_dict(
                {"techs": techs_this_node_incl_inheritance},
                MODEL_SCHEMA,
                f"tech definition at node `{node_name}`",
            )
            self._raise_error_on_transmission_tech_def(
                techs_this_node_incl_inheritance, node_name
            )
            techs_this_node_incl_inheritance.union(
                links_at_nodes.get(node_name, AttrDict())
            )

            tech_ds = self._definition_dict_to_ds(
                techs_this_node_incl_inheritance, "techs"
            )

            tech_ds.coords["nodes"] = node_name
            for ref_var in node_ref_vars:
                tech_ds[ref_var] = tech_ds[ref_var].expand_dims("nodes")
            for ref_var in ["carrier_in", "carrier_out"]:
                if ref_var in tech_ds.data_vars:
                    tech_ds[ref_var] = tech_ds[ref_var].expand_dims("nodes")
            if not tech_ds.nodes.shape:
                tech_ds["nodes"] = tech_ds["nodes"].expand_dims("nodes")

            node_tech_data.append(tech_ds)

        node_tech_ds = xr.combine_nested(
            node_tech_data,
            concat_dim="nodes",
            data_vars="minimal",
            combine_attrs="no_conflicts",
            coords="minimal",
        )

        validate_dict(
            {"nodes": active_node_dict}, MODEL_SCHEMA, "node (non-tech) definition"
        )
        self.model_data, node_tech_ds
        node_ds = self._definition_dict_to_ds(active_node_dict, "nodes")
        ds = xr.merge([node_tech_ds, node_ds])

        self.model_data = xr.merge(
            [ds, self.model_data], compat="override", combine_attrs="no_conflicts"
        ).fillna(self.model_data)

    def add_top_level_params(self):
        """Process any parameters defined in the top-level `parameters` key.

        Raises:
            KeyError: Cannot provide the same name for a top-level parameter as those defined already at the tech/node level.

        """
        if "parameters" not in self.model_definition:
            return None
        for param_name, param_data in self.model_definition["parameters"].items():
            if param_name in self.model_data.data_vars:
                raise KeyError(
                    f"Trying to add top-level parameter with same name as a node/tech level parameter: {param_name}"
                )
            param_dict = self._prepare_param_dict(param_name, param_data)
            param_da = self._param_dict_to_array(param_name, param_dict)
            self._update_param_coords(param_name, param_da)
            self._log_param_updates(param_name, param_da)
            param_ds = param_da.to_dataset()

            if "techs" in param_da.dims and "nodes" in param_da.dims:
                valid_node_techs = (
                    param_da.to_series().dropna().groupby(["nodes", "techs"]).first()
                )
                exceptions.warn(
                    f"(parameters, {param_name}) | This parameter will only take effect if you have already defined"
                    f" the following combinations of techs at nodes in your model definition: {valid_node_techs.index.values}"
                )

            self.model_data = self.model_data.merge(param_ds)

    def add_time_dimension(self, timeseries_dfs: Optional[dict[str, pd.DataFrame]]):
        """Process file/dataframe references in the model data and use it to expand the model to include a time dimension.

        Args:
            timeseries_dfs (Optional[dict[str, pd.DataFrame]]):
                Reference to pre-loaded pandas.DataFrame objects, if any reference to them in the model definition (`via df=`).

        Raises:
            exceptions.ModelError: The model has to have a time dimension, so at least one reference must exist.
        """
        self.model_data = time.add_time_dimension(
            self.model_data, self.config, timeseries_dfs
        )
        if "timesteps" not in self.model_data:
            raise exceptions.ModelError(
                "Must define at least one timeseries parameter in a Calliope model."
            )
        self.model_data = time.add_inferred_time_params(self.model_data)

    def resample_time_dimension(self):
        """If resampling/clustering is requested in the initialisation config, apply it here."""
        if self.config["time_resample"] is not None:
            self.model_data = time.resample(
                self.model_data, self.config["time_resample"]
            )
        if self.config["time_cluster"] is not None:
            self.model_data = time.cluster(
                self.model_data, self.config["time_cluster"], self.config["time_format"]
            )

    def clean_data_from_undefined_members(self):
        """Generate the `definition_matrix` array and use it to strip out any dimension items that are NaN in all arrays and any arrays that are NaN in all index positions."""
        def_matrix = (
            self.model_data.carrier_in.notnull() | self.model_data.carrier_out.notnull()
        )
        # NaNing values where they are irrelevant requires definition_matrix to be boolean
        for var_name, var_data in self.model_data.data_vars.items():
            non_dims = set(def_matrix.dims).difference(var_data.dims)
            self.model_data[var_name] = var_data.where(def_matrix.any(non_dims))

        # dropping index values where they are irrelevant requires definition_matrix to be NaN where False
        self.model_data["definition_matrix"] = def_matrix.where(def_matrix)
        for dim in def_matrix.dims:
            orig_dim_vals = set(self.model_data.coords[dim].data)
            self.model_data = self.model_data.dropna(
                dim, how="all", subset=["definition_matrix"]
            )
            deleted_dim_vals = orig_dim_vals.difference(
                set(self.model_data.coords[dim].data)
            )
            if deleted_dim_vals:
                LOGGER.debug(
                    f"Deleting {dim} values as they are not defined anywhere in the model: {deleted_dim_vals}"
                )

        # The boolean version of definition_matrix is what we keep
        self.model_data["definition_matrix"] = def_matrix

        vars_to_delete = [
            var_name
            for var_name, var in self.model_data.data_vars.items()
            if var.isnull().all()
        ]
        if vars_to_delete:
            LOGGER.debug(f"Deleting empty parameters: {vars_to_delete}")
        self.model_data = self.model_data.drop_vars(vars_to_delete)

    def add_link_distances(self):
        """If latitude/longitude are provided but distances between nodes have not been computed, compute them now.

        The schema will have already handled the fact that if one of lat/lon is provided, the other must also be provided.

        """
        # If no distance was given, we calculate it from coordinates
        if (
            "latitude" in self.model_data.data_vars
            and "longitude" in self.model_data.data_vars
        ):
            geod = geodesic.Geodesic.WGS84
            distances = {}
            for tech in self.model_data.techs:
                if self.model_data.parent.sel(techs=tech).item() != "transmission":
                    continue
                tech_def = self.model_data.definition_matrix.sel(techs=tech).any(
                    "carriers"
                )
                node1, node2 = tech_def.where(tech_def).dropna("nodes").nodes.values
                distances[tech.item()] = geod.Inverse(
                    self.model_data.latitude.sel(nodes=node1).item(),
                    self.model_data.longitude.sel(nodes=node1).item(),
                    self.model_data.latitude.sel(nodes=node2).item(),
                    self.model_data.longitude.sel(nodes=node2).item(),
                )["s12"]
            distance_array = pd.Series(distances).rename_axis(index="techs").to_xarray()
            if self.config["distance_unit"] == "km":
                distance_array /= 1000
        else:
            LOGGER.debug(
                "Link distances will not be computed automatically since lat/lon coordinates are not defined."
            )
            return None

        if "distance" not in self.model_data.data_vars:
            self.model_data["distance"] = distance_array
            LOGGER.debug(
                "Link distance matrix automatically computed from lat/lon coordinates."
            )
        else:
            self.model_data["distance"] = self.model_data["distance"].fillna(
                distance_array
            )
            LOGGER.debug(
                "Any missing link distances automatically computed from lat/lon coordinates."
            )

    def add_colors(self):
        """If technology colours have not been provided / only partially provided, generate a sequence of colors to fill the gap.

        This is a convenience function for downstream plotting.
        Since we have removed core plotting components from Calliope, it is not a strictly necessary preprocessing step.
        """
        techs = self.model_data.techs
        color_array = self.model_data.get("color")
        default_palette_cycler = itertools.cycle(range(len(self._DEFAULT_PALETTE)))
        new_color_array = xr.DataArray(
            [self._DEFAULT_PALETTE[next(default_palette_cycler)] for tech in techs],
            coords={"techs": techs},
        )
        if color_array is None:
            LOGGER.debug("Building technology color array from default palette.")
            self.model_data["color"] = new_color_array
        elif color_array.isnull().any():
            LOGGER.debug(
                "Filling missing technology color array values from default palette."
            )
            self.model_data["color"] = self.model_data["color"].fillna(new_color_array)

    def assign_input_attr(self):
        """All input parameters need to be assigned the `is_result=False` attribute to be able to filter the arrays in the calliope.Model object."""
        for var_name, var_data in self.model_data.data_vars.items():
            self.model_data[var_name] = var_data.assign_attrs(is_result=False)

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
        self._tech_data_source_dict.union(base_tech_data, allow_override=True)

        tech_dict = AttrDict({k: {} for k in techs.values})
        tech_dict.union(self.model_definition["techs"], allow_override=True)
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
        techs_incl_inheritance = self._inherit_defs("techs")
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
                if techs_incl_inheritance[tech].get("parent", None) == "transmission":
                    if "from" in self._tech_data_source_dict[tech]:
                        self._tech_data_source_dict.set_key(f"{tech}.to", node)
                    else:
                        self._tech_data_source_dict.set_key(f"{tech}.from", node)
                else:
                    tech_dict["techs"][tech] = None

        node_tech_dict.union(self.model_definition["nodes"], allow_override=True)
        self.model_definition["nodes"] = node_tech_dict

    def _get_relevant_node_refs(self, techs_dict: AttrDict, node: str) -> list[str]:
        """Get all references to parameters made in technologies at nodes.

        This defines those arrays in the dataset that *must* be indexed over `nodes` as well as `techs`.

        If timeseries files/dataframes are referenced in a tech at a node, the node name is added as the column name in-place.
        Techs *must* define these timeseries references explicitly at nodes to access different data columns at different nodes.

        Args:
            techs_dict (AttrDict): Dictionary of technologies defined at a node.
            node (str): Name of the node.

        Returns:
            list[str]: List of parameters at this node that must be indexed over the node dimension.
        """
        refs = set()
        for key, val in techs_dict.as_dict_flat().items():
            if (
                isinstance(val, str)
                and val.startswith(("file=", "df="))
                and ":" not in val
            ):
                techs_dict.set_key(key, val + ":" + node)

        for tech_name, tech_dict in techs_dict.items():
            if tech_dict is None or not tech_dict.get("active", True):
                if isinstance(tech_dict, dict) and not tech_dict.get("active", True):
                    self._deactivate_item(techs=tech_name, nodes=node)
                continue
            else:
                if "parent" in tech_dict.keys():
                    raise exceptions.ModelError(
                        f"(nodes, {node}), (techs, {tech_name}) | Defining a technology `parent` at a node is not supported; "
                        "limit yourself to defining this parameter within `techs` or `tech_groups`"
                    )
                refs.update(tech_dict.keys())

        return list(refs)

    def _param_dict_to_array(self, param_name: str, param_data: Param) -> xr.DataArray:
        """Take a blessed parameter dictionary and convert it to an xarray DataArray.

        Args:
            param_name (str): Name of the parameter being converted.
            param_data (Param): Blessed dictionary. I.e., keys/values follow an expected structure.

        Returns:
            xr.DataArray: Array representation of the parameter.
        """
        if param_data["dims"]:
            param_series = pd.Series(
                data=param_data["data"],
                index=[tuple(idx) for idx in param_data["index"]],
            )
            param_series.index = pd.MultiIndex.from_tuples(
                param_series.index, names=param_data["dims"]
            )
            param_da = param_series.to_xarray()
        else:
            param_da = xr.DataArray(param_data["data"])
        param_da = param_da.rename(param_name).assign_attrs(
            self.param_attrs.get(param_name, {})
        )
        return param_da

    def _definition_dict_to_ds(
        self,
        def_dict: dict[str, dict[str, dict | list[str] | DATA_T]],
        dim_name: Literal["nodes", "techs"],
    ) -> xr.Dataset:
        """Convert a dictionary of nodes/techs with their parameter definitions into an xarray dataset.

        Node/tech name will be injected into each parameter's `index` and `dims` lists so that the resulting arrays include those dimensions.

        Args:
            def_dict (dict[str, dict[str, dict | list[str] | DATA_T]]):
                `node`/`tech` definitions.
                The first set of keys are dimension index items, the second set of keys are parameter names.
                Parameters need not be blessed.
            dim_name (Literal[nodes, techs]): Dimension name of the dictionary items.

        Returns:
            xr.Dataset: Dataset with arrays indexed over (at least) the input `dim_name`.
        """
        param_ds = xr.Dataset()
        for idx_name, idx_params in def_dict.items():
            param_das: list[xr.DataArray] = []
            for param_name, param_data in idx_params.items():
                param_dict = self._prepare_param_dict(param_name, param_data)
                param_dict["index"] = [[idx_name] + idx for idx in param_dict["index"]]
                param_dict["dims"].insert(0, dim_name)
                param_das.append(self._param_dict_to_array(param_name, param_dict))
            param_ds = xr.merge([param_ds, xr.combine_by_coords(param_das)])

        return param_ds

    def _prepare_param_dict(
        self, param_name: str, param_data: dict | list[str] | DATA_T
    ) -> Param:
        """Convert a range of parameter definitions into the blessed `Param` format, i.e.:

        ```
        data: numeric/boolean/string data or list of them.
        index: list of lists containing dimension index items (number of items in the sub-lists == length of `dims`).
        dims: list of dimension names.
        ```

        Args:
            param_name (str): Parameter name (used only in error messages).
            param_data (dict | list[str] | DATA_T): Input unformatted parameter data.

        Raises:
            ValueError: If the parameter is unindexed (i.e., no `dims`/`index`) and is not a lookup array (see LOOKUP_PARAMS), it cannot define a list of data.

        Returns:
            Param: Blessed parameter dictionary.
        """
        if isinstance(param_data, dict):
            data = param_data["data"]
            index_items = [listify(idx) for idx in listify(param_data["index"])]
            dims = listify(param_data["dims"])
        elif param_name in self.LOOKUP_PARAMS.keys():
            data = True
            index_items = [[i] for i in listify(param_data)]
            dims = [self.LOOKUP_PARAMS[param_name]]
        else:
            if isinstance(param_data, list):
                raise ValueError(
                    f"{param_name} | Cannot pass parameter data as a list unless the parameter is one of the pre-defined lookup arrays: {list(self.LOOKUP_PARAMS.keys())}."
                )
            data = param_data
            index_items = [[]]
            dims = []
        data_dict: Param = {"data": data, "index": index_items, "dims": dims}
        return data_dict

    def _inherit_defs(
        self,
        dim_name: Literal["nodes", "techs"],
        dim_dict: Optional[AttrDict] = None,
        err_message_prefix: str = "",
    ) -> AttrDict:
        """For a set of node/tech definitions, climb the inheritance tree to build a final definition dictionary.

        For `techs` at `nodes`, the first step is to inherit the technology definition from `techs`, _then_ to climb `inherit` references.

        Base definitions will take precedence over inherited ones and more recent inherited definitions will take precedence over older ones.

        If a `tech`/`node` has the `active` parameter set to `False` (including if it inherits this parameter), it will not make it into the output dictionary.

        Args:
            dim_name (Literal[nodes, techs]): Name of dimension we're working with.
            dim_dict (Optional[AttrDict], optional):
                Base dictionary to work from.
                If not defined, `dim_name` will be used to access the dictionary from the base model definition.
                Defaults to None.
            err_message_prefix (str, optional):
                If working with techs at nodes, it is prudent to provide the node name to prefix error messages. Defaults to "".

        Raises:
            KeyError: Cannot define a `tech` at a `node` if it isn't already defined under the `techs` top-level key.

        Returns:
            AttrDict: Dictionary containing all active tech/node definitions with inherited parameters.
        """
        updated_defs = AttrDict()
        if dim_dict is None:
            dim_dict = self.model_definition[dim_name]

        for item_name, item_def in dim_dict.items():
            if item_def is None:
                item_def = AttrDict()
            if dim_name == "techs":
                base_def = self.model_definition["techs"]
                if item_name not in base_def:
                    raise KeyError(
                        f"{err_message_prefix}({dim_name}, {item_name}) | Reference to item not defined in base {dim_name}"
                    )

                item_base_def = deepcopy(base_def[item_name])
                item_base_def.union(item_def, allow_override=True)
            else:
                item_base_def = item_def
            updated_item_def, inheritance = self._climb_inheritance_tree(
                item_base_def, dim_name, item_name
            )
            if dim_name == "techs" and item_name in self._tech_data_source_dict:
                _data_source_dict = self._tech_data_source_dict[item_name]
                _data_source_dict.union(updated_item_def, allow_override=True)
                updated_item_def = _data_source_dict

            if not updated_item_def.get("active", True):
                LOGGER.debug(
                    f"{err_message_prefix}({dim_name}, {item_name}) | Deactivated."
                )
                self._deactivate_item(**{dim_name: item_name})
                continue

            if inheritance is not None:
                updated_item_def[f"{dim_name}_inheritance"] = ",".join(inheritance)
                del updated_item_def["inherit"]

            updated_defs[item_name] = updated_item_def

        return updated_defs

    def _climb_inheritance_tree(
        self,
        dim_item_dict: AttrDict,
        dim_name: Literal["nodes", "techs"],
        item_name: str,
        inheritance: Optional[list] = None,
    ) -> tuple[AttrDict, Optional[list]]:
        """Follow the `inherit` references from `nodes` to `node_groups` / from `techs` to `tech_groups`.

        Abstract group definitions (those in `node_groups`/`tech_groups`) can inherit each other, but `nodes`/`techs` cannot.

        This function will be called recursively until a definition dictionary without `inherit` is reached.

        Args:
            dim_item_dict (AttrDict):
                Dictionary (possibly) containing `inherit`. If it doesn't contain `inherit`, the climbing stops here.
            dim_name (Literal[nodes, techs]):
                The name of the dimension we're working with, so that we can access the correct `_groups` definitions.
            item_name (str):
                The current position in the inheritance tree.
            inheritance (Optional[list], optional):
                A list of items that have been inherited (starting with the oldest).
                If the first `dim_item_dict` does not contain `inherit`, this will remain as None.
                Defaults to None.

        Raises:
            KeyError: Must inherit from a named group item in `node_groups` (for `nodes`) and `tech_groups` (for `techs`)

        Returns:
            tuple[AttrDict, Optional[list]]: Definition dictionary with inherited data and a list of the inheritance tree climbed to get there.
        """
        dim_name_singular = dim_name.removesuffix("s")
        dim_group_def = self.model_definition.get(f"{dim_name_singular}_groups", None)
        to_inherit = dim_item_dict.get("inherit", None)
        if to_inherit is None:
            updated_dim_item_dict = dim_item_dict
        elif dim_group_def is None or to_inherit not in dim_group_def:
            raise KeyError(
                f"({dim_name}, {item_name}) | Cannot find `{to_inherit}` in inheritance tree."
            )
        else:
            base_def_dict, inheritance = self._climb_inheritance_tree(
                dim_group_def[to_inherit], dim_name, to_inherit, inheritance
            )
            updated_dim_item_dict = deepcopy(base_def_dict)
            updated_dim_item_dict.union(dim_item_dict, allow_override=True)
            if inheritance is not None:
                inheritance.append(to_inherit)
            else:
                inheritance = [to_inherit]
        return updated_dim_item_dict, inheritance

    def _deactivate_item(self, **item_ref):
        for dim_name, item_name in item_ref.items():
            if item_name not in self.model_data.coords.get(dim_name, xr.DataArray()):
                return None
        if len(item_ref) == 1:
            self.model_data = self.model_data.drop_sel(**item_ref)
        else:
            if "carrier_in" in self.model_data:
                self.model_data["carrier_in"].loc[item_ref] = np.nan
            if "carrier_out" in self.model_data:
                self.model_data["carrier_out"].loc[item_ref] = np.nan

    def _links_to_node_format(self, active_node_dict: AttrDict) -> AttrDict:
        """Process `transmission` techs into links by assigned them to the nodes defined by their `from` and `to` keys.

        Args:
            active_node_dict (AttrDict):
                Dictionary of nodes that are active in this model.
                If a transmission tech references a non-active / undefined node, a link will not be generated.

        Returns:
            AttrDict: Dictionary of transmission techs distributed to nodes (of the form {node_name: {tech_name: {...}, tech_name: {}}}).
        """
        active_link_techs = AttrDict(
            {
                tech: tech_def
                for tech, tech_def in self._inherit_defs("techs").items()
                if tech_def.get("parent") == "transmission"
            }
        )
        validate_dict(
            {"techs": active_link_techs}, MODEL_SCHEMA, "link tech definition"
        )
        link_tech_dict = AttrDict()
        if not active_link_techs:
            LOGGER.debug("links | No links between nodes defined.")

        for link_name, link_data in active_link_techs.items():
            # TODO: fix issue of links defined in data sources not ending up in this loop
            # and therefore not having the chance to be deactivated when one / both connecting nodes are deactivated.
            node_from, node_to = link_data.pop("from"), link_data.pop("to")
            nodes_exists = all(
                node in active_node_dict
                or node_from in self.model_data.coords.get("nodes", xr.DataArray())
                for node in [node_from, node_to]
            )
            if not nodes_exists:
                LOGGER.debug(
                    f"(links, {link_name}) | Deactivated due to missing/deactivated `from` or to `node`."
                )
                self._deactivate_item(techs=link_name)
                continue
            node_from_data = link_data.copy()
            node_to_data = link_data.copy()

            if link_data.get("one_way", False):
                self._update_one_way_links(node_from_data, node_to_data)

            link_tech_dict.union(
                AttrDict(
                    {
                        node_from: {link_name: node_from_data},
                        node_to: {link_name: node_to_data},
                    }
                )
            )

        return link_tech_dict

    def _update_param_coords(self, param_name: str, param_da: xr.DataArray):
        """
        Check array coordinates to see if any should be in datetime format,
        if the base model coordinate is in datetime format.

        Args:
            param_name (str): name of parameter being added to the model.
            param_da (xr.DataArray): array of parameter data.
        """

        to_update = {}
        for coord_name, coord_data in param_da.coords.items():
            coord_in_model = coord_name in self.model_data.coords
            if coord_in_model and self.model_data[coord_name].dtype.kind == "M":
                to_update[coord_name] = pd.to_datetime(coord_data, format="ISO8601")
            elif not coord_in_model:
                try:
                    to_update[coord_name] = pd.to_datetime(coord_data, format="ISO8601")
                except ValueError:
                    continue
        for coord_name, coord_data in to_update.items():
            param_da.coords[coord_name] = coord_data
            LOGGER.debug(
                f"(parameters, {param_name}) | Updating {coord_name} dimension index values to datetime format"
            )

    def _log_param_updates(self, param_name: str, param_da: xr.DataArray):
        """
        Check array coordinates to see if:
            1. any are new compared to the base model dimensions.
            2. any are adding new elements to an existing base model dimension.

        Args:
            param_name (str): name of parameter being added to the model.
            param_da (xr.DataArray): array of parameter data.
        """
        for coord_name, coord_data in param_da.coords.items():
            if coord_name not in self.model_data.coords:
                LOGGER.debug(
                    f"(parameters, {param_name}) | Adding a new dimension to the model: {coord_name}"
                )
            else:
                new_coord_data = coord_data[
                    ~coord_data.isin(self.model_data.coords[coord_name])
                ]
                if new_coord_data.size > 0:
                    LOGGER.debug(
                        f"(parameters, {param_name}) | Adding a new value to the "
                        f"`{coord_name}` model coordinate: {new_coord_data.values}"
                    )

    @staticmethod
    def _listify_if_defined(source: DataSource, key: str) -> Optional[list]:
        vals = source.get(key, None)
        if vals is not None:
            vals = listify(vals)
        return vals

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
            carriers = set(
                grouped_series.dropna(subset=[f"carrier_{direction}"]).carriers.values
            )
            if len(carriers) == 1:
                carriers = carriers.pop()
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
    def _update_one_way_links(node_from_data: dict, node_to_data: dict):
        """For one-way transmission links, delete option to have carrier outflow (imports) at the `from` node and carrier inflow (exports) at the `to` node.

        Deletions happen on the tech definition dictionaries in-place.

        Args:
            node_from_data (dict): Link technology data dictionary at the `from` node.
            node_to_data (dict): Link technology data dictionary at the `to` node.
        """
        node_from_data.pop("carrier_out")  # cannot import carriers at the `from` node
        node_to_data.pop("carrier_in")  # cannot export carrier at the `to` node

    def _raise_error_on_transmission_tech_def(
        self, tech_def_dict: AttrDict, node_name: str
    ):
        """Do not allow any transmission techs are defined in the node-level tech dict.

        Args:
            tech_def_dict (dict): Tech definition dict (after full inheritance) at a node.
            node_name (str): Node name.

        Raises:
            exceptions.ModelError: Raise if any defined techs have the `transmission` parent.
        """
        transmission_techs = [
            k for k, v in tech_def_dict.items() if v.get("parent", "") == "transmission"
        ]

        if transmission_techs:
            raise exceptions.ModelError(
                f"(nodes, {node_name}) | Transmission techs cannot be directly defined at nodes; "
                f"they will be automatically assigned to nodes based on `to` and `from` parameters: {transmission_techs}"
            )
