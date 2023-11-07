import itertools
import logging
from copy import deepcopy
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
from geographiclib import geodesic
from typing_extensions import NotRequired, TypedDict

from calliope.attrdict import AttrDict
from calliope.preprocess import time
from calliope.util import tools

LOGGER = logging.getLogger(__name__)

DATA_T = Optional[float | int | bool | str] | list[Optional[float | int | bool | str]]


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


LOGGER = logging.getLogger(__name__)


class ModelDataFactory:
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
        model_def_schema: dict,
    ):
        """
        Take a Calliope model_run and convert it into an xarray Dataset, ready for
        constraint generation. Timeseries data is also extracted from file at this
        point, and the time dimension added to the data

        Parameters
        ----------
        model_run_dict : AttrDict
            preprocessed model_run dictionary, as produced by
            Calliope.preprocess.preprocess_model

        Returns
        -------
        data : xarray Dataset
            Dataset with optimisation param_dict as variables, optimisation sets as
            coordinates, and other information in attributes.
        data_pre_time : xarray Dataset, only returned if debug = True
            Dataset, prior to time dimension addition, with optimisation param_dict
            as variables, optimisation sets as coordinates, and other information
            in attributes.

        """
        self.config: dict = model_config
        self.model_definition: ModelDefinition = model_definition.copy()
        self.schema = model_def_schema
        self.model_data = xr.Dataset(attrs=AttrDict({**attributes, **param_attributes}))

        flipped_attributes: dict[str, dict] = dict()
        for key, val in param_attributes.items():
            for subkey, subval in val.items():
                flipped_attributes.setdefault(subkey, {})
                flipped_attributes[subkey][key] = subval
        self.param_attrs = flipped_attributes

    def add_node_tech_data(self):
        active_node_dict = self._inherit_defs("nodes")
        links_at_nodes = self._links_to_node_format(active_node_dict)

        node_tech_data = []
        for node_name, node_data in active_node_dict.items():
            techs_this_node = node_data.pop("techs")
            if techs_this_node is None:
                techs_this_node = AttrDict()
            node_ref_vars = self._get_relevant_node_refs(techs_this_node, node_name)
            techs_this_node_incl_inheritance = self._inherit_defs(
                "techs",
                techs_this_node,
                err_message_prefix=f"(nodes, {node_name}), ",
            )

            tools.validate_dict(
                {"techs": techs_this_node_incl_inheritance},
                self.schema,
                f"tech definition at node `{node_name}`",
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
            tech_ds = self._add_active_node_tech(tech_ds)
            node_tech_data.append(tech_ds)

        node_tech_ds = xr.combine_nested(
            node_tech_data,
            concat_dim="nodes",
            data_vars="different",
            combine_attrs="no_conflicts",
            coords="different",
        )

        node_ds = self._definition_dict_to_ds(active_node_dict, "nodes")
        self.model_data = xr.merge([self.model_data, node_tech_ds, node_ds])

    def add_top_level_params(self):
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
                param_ds = self._add_active_node_tech(param_ds)

            self.model_data = self.model_data.merge(param_ds)

    def add_time_dimension(self, timeseries_dfs: Optional[dict[str, pd.DataFrame]]):
        self.model_data = time.add_time_dimension(
            self.model_data, self.config, timeseries_dfs
        )
        self.model_data = time.add_inferred_time_params(self.model_data)
        if self.config["time_resample"] is not None:
            self.model_data = time.resample(
                self.model_data, self.config["time_resample"]
            )
        if self.config["time_cluster"] is not None:
            self.model_data = time.cluster(self.model_data, self.config["time_cluster"])

        self._update_dtypes()

    def clean_data_from_undefined_members(self):
        def_matrix = self.model_data.active.notnull() & (
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
        # If no distance was given, we calculate it from coordinates
        if (
            "latitude" in self.model_data.data_vars
            and "longitude" in self.model_data.data_vars
        ):
            geod = geodesic.Geodesic.WGS84
            distances = {}
            for tech in self.model_data.techs:
                latitudes = self.model_data.latitude.sel(techs=tech).dropna("nodes")
                longitudes = self.model_data.longitude.sel(techs=tech).dropna("nodes")
                node1, node2 = latitudes.nodes.values
                distances[tech] = geod.Inverse(
                    latitudes.sel(nodes=node1),
                    longitudes.sel(nodes=node1),
                    latitudes.sel(nodes=node2),
                    longitudes.sel(nodes=node2),
                )["s12"]
            distance_array = pd.Series(distances).rename_axis(index="techs").to_xarray()
        else:
            distance_array = xr.DataArray(np.nan)

        if "distance" not in self.model_data.data_vars:
            self.model_data["distance"] = distance_array
            LOGGER.debug(
                "Link distance matrix automatically computed from lat/lon coordinates."
            )
        else:
            self.model_data["distance"].fillna(distance_array)
            LOGGER.debug(
                "Missing link distances automatically computed from lat/lon coordinates."
            )

    def assign_input_attr(self):
        for var_name, var_data in self.model_data.data_vars.items():
            self.model_data[var_name] = var_data.assign_attrs(is_result=False)

    def _get_relevant_node_refs(self, tech_dict: AttrDict, node: str):
        refs = set()
        for key, val in tech_dict.as_dict_flat().items():
            if (
                isinstance(val, str)
                and val.startswith(("file=", "df="))
                and ":" not in val
            ):
                tech_dict.set_key(key, val + ":" + node)

        for tech, _dict in tech_dict.items():
            if _dict is None:
                continue
            else:
                refs.update(_dict.keys())
        return list(refs)

    @staticmethod
    def _add_active_node_tech(ds: xr.Dataset) -> xr.Dataset:
        if not ds.nodes.shape:
            ds["nodes"] = ds["nodes"].expand_dims("nodes")

        ds["active"] = xr.DataArray(
            data=True,
            dims=("nodes", "techs"),
            coords={"nodes": ds.nodes, "techs": ds.techs},
        )
        return ds

    def _param_dict_to_array(self, param_name: str, param_data: Param) -> xr.DataArray:
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

    def _combine_param_dicts_to_dataset(self, params: dict[str, Param]) -> xr.Dataset:
        param_ds = xr.combine_by_coords(
            [
                self._param_dict_to_array(param_name, param_data)
                for param_name, param_data in params.items()
            ]
        )
        return param_ds

    def _definition_dict_to_ds(
        self,
        dim_dict: dict[str, dict[str, dict | list[str] | DATA_T]],
        dim_name: str,
    ) -> xr.Dataset:
        param_ds = xr.Dataset()
        for idx_name, idx_params in dim_dict.items():
            params: dict[str, Param] = {}
            for param_name, param_data in idx_params.items():
                param_dict = self._prepare_param_dict(param_name, param_data)
                param_dict["index"] = [[idx_name] + idx for idx in param_dict["index"]]
                param_dict["dims"].insert(0, dim_name)
                params[param_name] = param_dict
            param_ds = xr.merge(
                [param_ds, self._combine_param_dicts_to_dataset(params)]
            )

        return param_ds

    def _prepare_param_dict(
        self, param_name: str, param_data: dict | list[str] | DATA_T
    ) -> Param:
        if isinstance(param_data, dict):
            data = param_data["data"]
            index_items = [
                tools.listify(idx) for idx in tools.listify(param_data["index"])
            ]
            dims = tools.listify(param_data["dims"])
        elif param_name in self.LOOKUP_PARAMS.keys():
            data = True
            index_items = [[i] for i in tools.listify(param_data)]
            dims = [self.LOOKUP_PARAMS[param_name]]
        else:
            data = param_data
            index_items = [[]]
            dims = []
        return {
            "data": data,
            "index": index_items,
            "dims": dims,
        }

    def _inherit_defs(
        self,
        dim_name: Literal["nodes", "techs"],
        dim_dict: Optional[AttrDict] = None,
        err_message_prefix: str = "",
    ) -> AttrDict:
        updated_defs = AttrDict()
        if dim_dict is None:
            dim_dict = self.model_definition[dim_name]

        for item_name, item_def in dim_dict.items():
            if item_def is None:
                item_def = AttrDict()
            if not item_def.get("active", True):
                LOGGER.debug(f"({dim_name}, {item_name}) | Deactivated.")
                continue
            if dim_name == "techs":
                base_def = self.model_definition["techs"]
                if item_name not in base_def:
                    raise KeyError(
                        f"{err_message_prefix}({dim_name}, {item_name}) | Reference to item not defined in {dim_name}"
                    )

                item_base_def = deepcopy(base_def[item_name])
                item_base_def.union(item_def, allow_override=True)
            else:
                item_base_def = item_def
            updated_defs[item_name], inheritance = self._climb_inheritance_tree(
                item_base_def, dim_name, item_name
            )
            if inheritance is not None:
                updated_defs[item_name][f"{dim_name}_inheritance"] = ",".join(
                    inheritance
                )
                updated_defs.del_key(f"{item_name}.inherit")
        return updated_defs

    def _climb_inheritance_tree(
        self,
        dim_item_dict: AttrDict,
        dim_name: str,
        item_name: str,
        inheritance: Optional[list] = None,
    ) -> tuple[AttrDict, Optional[list]]:
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
                dim_group_def[to_inherit], dim_name_singular, to_inherit, inheritance
            )
            updated_dim_item_dict = deepcopy(base_def_dict)
            updated_dim_item_dict.union(dim_item_dict, allow_override=True)
            if inheritance is not None:
                inheritance.append(to_inherit)
            else:
                inheritance = [to_inherit]
        return updated_dim_item_dict, inheritance

    @staticmethod
    def _duplicate_link_techs(link_name, link_tech_def, linked_nodes):
        new_tech_dict = {
            node: {link_name: v for k, v in link_tech_def.items()}
            for node in linked_nodes
        }
        return new_tech_dict

    def _links_to_node_format(self, active_node_dict: AttrDict) -> AttrDict:
        active_link_techs = AttrDict(
            {
                tech: tech_def
                for tech, tech_def in self._inherit_defs("techs").items()
                if tech_def["parent"] == "transmission"
            }
        )
        tools.validate_dict(
            {"techs": active_link_techs},
            self.schema,
            "link tech definition",
        )
        link_tech_dict = AttrDict()
        if not active_link_techs:
            LOGGER.debug("links | No links between nodes defined.")

        for link_name, link_data in active_link_techs.items():
            node_from, node_to = link_data.pop("from"), link_data.pop("to")
            if not any(node in active_node_dict for node in [node_from, node_to]):
                LOGGER.debug(
                    f"(links, {link_name}) | Deactivated due to missing/deactivated `from` or to `node`."
                )
                continue

            link_tech_dict.union(
                AttrDict(
                    {node: {link_name: link_data} for node in [node_from, node_to]}
                )
            )

        return link_tech_dict

    def add_colors(self) -> xr.DataArray:
        techs = self.model_data.techs
        color_array = self.model_data.get("color")
        default_palette_cycler = itertools.cycle(range(len(self._DEFAULT_PALETTE)))
        new_color_array = xr.DataArray(
            [self._DEFAULT_PALETTE[next(default_palette_cycler)] for tech in techs],
            coords={"techs": techs},
        )
        if color_array is None:
            LOGGER.debug("Building technology color array from default palette.")
            return new_color_array
        elif color_array.isnull().any():
            LOGGER.debug(
                "Filling missing technology color array values from default palette."
            )
            return color_array.fillna(new_color_array)
        else:
            return color_array

    def _update_param_coords(self, param_name: str, param_da: xr.DataArray) -> None:
        """
        Check array coordinates to see if any should be in datetime format,
        if the base model coordinate is in datetime format.

        Args:
            param_name (str): name of parameter being added to the model.
            param_da (xr.DataArray): array of parameter data.
        """
        coords_to_update = {}
        for coord_name, coord_data in param_da.coords.items():
            if self.model_data.coords.get(coord_name, xr.DataArray()).dtype.kind == "M":
                LOGGER.debug(
                    f"(parameters, {param_name}) | Updating {coord_name} dimension index values to datetime format"
                )
                coords_to_update[coord_name] = pd.to_datetime(
                    coord_data, format="ISO8601"
                )
        for coord_name, coord_data in coords_to_update.items():
            param_da.coords[coord_name] = coord_data

    def _log_param_updates(self, param_name: str, param_da: xr.DataArray) -> None:
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

    def _update_dtypes(self):
        """
        Update dtypes to not be 'Object', if possible.
        Order of preference is: bool, int, float
        """
        # TODO: this should be redundant once we load data from file and can check types from the schema
        for var_name, var in self.model_data.data_vars.items():
            if var.dtype.kind == "O":
                no_nans = var.where(var != "nan", drop=True)
                self.model_data[var_name] = var.where(var != "nan")
                if no_nans.isin(["True", 0, 1, "False", "0", "1"]).all():
                    # Turn to bool
                    self.model_data[var_name] = var.isin(["True", 1, "1"])
                else:
                    try:
                        self.model_data[var_name] = var.astype(np.int_, copy=False)
                    except (ValueError, OverflowError):
                        try:
                            self.model_data[var_name] = var.astype(
                                np.float_, copy=False
                            )
                        except ValueError:
                            None
