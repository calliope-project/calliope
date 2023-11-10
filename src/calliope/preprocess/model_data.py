import itertools
import logging
from copy import deepcopy
from typing import Literal, Optional

import pandas as pd
import xarray as xr
from geographiclib import geodesic
from typing_extensions import NotRequired, TypedDict

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.preprocess import time
from calliope.util.schema import MODEL_SCHEMA, validate_dict
from calliope.util.tools import listify

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
        self.model_data = xr.Dataset(attrs=AttrDict(attributes))

        flipped_attributes: dict[str, dict] = dict()
        for key, val in param_attributes.items():
            for subkey, subval in val.items():
                flipped_attributes.setdefault(subkey, {})
                flipped_attributes[subkey][key] = subval
        self.param_attrs = flipped_attributes

    def build(self, timeseries_dfs: Optional[dict[str, pd.DataFrame]]):
        self.add_node_tech_data()
        self.add_time_dimension(timeseries_dfs)
        self.add_top_level_params()
        self.clean_data_from_undefined_members()
        self.add_colors()
        self.add_link_distances()
        self.resample_time_dimension()
        self.assign_input_attr()

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
            validate_dict(
                {"techs": techs_this_node_incl_inheritance},
                MODEL_SCHEMA,
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
            data_vars="minimal",
            combine_attrs="no_conflicts",
            coords="minimal",
        )

        validate_dict(
            {"nodes": active_node_dict}, MODEL_SCHEMA, "node (non-tech) definition"
        )

        node_ds = self._definition_dict_to_ds(active_node_dict, "nodes")
        self.model_data = xr.merge([self.model_data, node_tech_ds, node_ds])

    def add_top_level_params(self):
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
                param_ds = self._add_active_node_tech(param_ds)

            self.model_data = self.model_data.merge(param_ds)

    def add_time_dimension(self, timeseries_dfs: Optional[dict[str, pd.DataFrame]]):
        self.model_data = time.add_time_dimension(
            self.model_data, self.config, timeseries_dfs
        )
        if "timesteps" not in self.model_data:
            raise exceptions.ModelError(
                "Must define at least one timeseries parameter in a Calliope model."
            )
        self.model_data = time.add_inferred_time_params(self.model_data)

    def resample_time_dimension(self):
        if self.config["time_resample"] is not None:
            self.model_data = time.resample(
                self.model_data, self.config["time_resample"]
            )
        if self.config["time_cluster"] is not None:
            self.model_data = time.cluster(
                self.model_data, self.config["time_cluster"], self.config["time_format"]
            )

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
        for var_name, var_data in self.model_data.data_vars.items():
            self.model_data[var_name] = var_data.assign_attrs(is_result=False)

    def _get_relevant_node_refs(self, techs_dict: AttrDict, node: str) -> list[str]:
        refs = set()
        for key, val in techs_dict.as_dict_flat().items():
            if (
                isinstance(val, str)
                and val.startswith(("file=", "df="))
                and ":" not in val
            ):
                techs_dict.set_key(key, val + ":" + node)

        for tech_dict in techs_dict.values():
            if tech_dict is None or not tech_dict.get("active", True):
                continue
            else:
                refs.update(tech_dict.keys())
        return list(refs)

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

    def _definition_dict_to_ds(
        self,
        def_dict: dict[str, dict[str, dict | list[str] | DATA_T]],
        dim_name: str,
    ) -> xr.Dataset:
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
        data_dict: Param = {
            "data": data,
            "index": index_items,
            "dims": dims,
        }
        return data_dict

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
            if not updated_item_def.get("active", True):
                LOGGER.debug(
                    f"{err_message_prefix}({dim_name}, {item_name}) | Deactivated."
                )
                continue

            if inheritance is not None:
                updated_item_def[f"{dim_name}_inheritance"] = ",".join(inheritance)
                del updated_item_def["inherit"]

            updated_defs[item_name] = updated_item_def

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

    def _links_to_node_format(self, active_node_dict: AttrDict) -> AttrDict:
        active_link_techs = AttrDict(
            {
                tech: tech_def
                for tech, tech_def in self._inherit_defs("techs").items()
                if tech_def.get("parent") == "transmission"
            }
        )
        validate_dict(
            {"techs": active_link_techs},
            MODEL_SCHEMA,
            "link tech definition",
        )
        link_tech_dict = AttrDict()
        if not active_link_techs:
            LOGGER.debug("links | No links between nodes defined.")

        for link_name, link_data in active_link_techs.items():
            node_from, node_to = link_data.pop("from"), link_data.pop("to")

            if any(node not in active_node_dict for node in [node_from, node_to]):
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

    def _update_param_coords(self, param_name: str, param_da: xr.DataArray) -> None:
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

    @staticmethod
    def _add_active_node_tech(ds: xr.Dataset) -> xr.Dataset:
        if not ds.nodes.shape:
            ds["nodes"] = ds["nodes"].expand_dims("nodes")
        if not ("techs" in ds.coords and "nodes" in ds.coords):
            return ds
        ds["active"] = xr.DataArray(
            data=True,
            dims=("nodes", "techs"),
            coords={"nodes": ds.nodes, "techs": ds.techs},
        )
        return ds
