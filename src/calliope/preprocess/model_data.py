# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
model_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying param_dict.

"""
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

import calliope
from calliope import exceptions
from calliope._version import __version__
from calliope.core.attrdict import AttrDict
from calliope.preprocess import checks, time

LOGGER = logging.getLogger(__name__)


class ModelDataFactory:
    UNWANTED_TECH_KEYS = [
        "allowed_constraints",
        "required_constraints",
        "allowed_costs",
        "allowed_switches",
        "constraints",
        "essentials.carrier",
    ]

    LOOKUP_STR = "[\\w\\-]*"  # all alphanumerics + `_` and `-`

    def __init__(self, model_run_dict):
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
        self.node_dict: dict = model_run_dict.nodes.as_dict_flat()
        self.tech_dict: dict = model_run_dict.techs.as_dict_flat()
        self.timeseries_vars: pd.DataFrame = model_run_dict.timeseries_vars
        self.timeseries_data: set[str] = model_run_dict.timeseries_data
        self.time_config: Optional[AttrDict] = model_run_dict.config.init.time
        self.random_seed: Optional[int] = model_run_dict.config.init.random_seed
        self.params: AttrDict = model_run_dict.parameters

        self.model_data = xr.Dataset(coords={"timesteps": self.timeseries_data.index})
        self._add_attributes(model_run_dict)
        self.template_config = AttrDict.from_yaml(
            os.path.join(
                os.path.dirname(calliope.__file__), "config", "model_data_lookup.yaml"
            )
        )
        self._strip_unwanted_keys()
        self._add_node_tech_sets()

    def __call__(self):
        self._extract_node_tech_data()
        self._add_time_dimension()
        self._clean_model_data()

        return (
            self.model_data_pre_clustering,
            self.model_data,
            self.data_pre_time,
            self.stripped_keys,
        )

    def _extract_node_tech_data(self):
        self._add_param_from_template()
        self._add_definition_matrix()
        self._clean_unused_techs_nodes_and_carriers()

    def _add_top_level_params(self):
        for param_name, param_data in self.params.items():
            if param_name in self.model_data.data_vars:
                raise KeyError(
                    f"Trying to add top-level parameter with same name as a node/tech level parameter: {param_name}"
                )
            if "dims" in param_data:
                param_series = pd.Series(
                    data=param_data["data"],
                    index=param_data["index"],
                    name=param_name,
                )
                if isinstance(param_data["dims"], list) and len(param_data["dims"]) > 1:
                    param_series.index = pd.MultiIndex.from_tuples(
                        param_series.index, names=param_data["dims"]
                    )
                else:
                    param_series = param_series.rename_axis(param_data["dims"])
                param_da = param_series.to_xarray()
            else:
                param_da = xr.DataArray(param_data["data"], name=param_name)

            coords_to_update = {}
            for coord_name, coord_data in param_da.coords.items():
                if (
                    self.model_data.coords.get(coord_name, xr.DataArray()).dtype.kind
                    == "M"
                ):
                    LOGGER.debug(
                        f"Updating `{param_name}` {coord_name} dimension index values to datetime format"
                    )
                    coords_to_update[coord_name] = pd.to_datetime(
                        coord_data, format="ISO8601"
                    )
            for coord_name, coord_data in coords_to_update.items():
                param_da.coords[coord_name] = coord_data

            for coord_name, coord_data in param_da.coords.items():
                if coord_name not in self.model_data.coords:
                    LOGGER.debug(
                        f"top-level parameter `{param_name}` is adding a new dimension to the model: {coord_name}"
                    )
                else:
                    new_coord_data = coord_data[
                        ~coord_data.isin(self.model_data.coords[coord_name])
                    ]
                    if new_coord_data.size > 0:
                        LOGGER.debug(
                            f"top-level parameter `{param_name}` is adding a new value to the "
                            f"`{coord_name}` model coordinate: {new_coord_data.values}"
                        )
            self.model_data = self.model_data.merge(param_da.to_dataset())

    def _add_time_dimension(self):
        self.data_pre_time = self.model_data.copy(deep=True)
        self.model_data = time.add_time_dimension(
            self.model_data,
            self.timeseries_vars,
            self.timeseries_data,
        )
        self._update_dtypes()

        # TODO: this is a legacy function, so should be updated to numpy best-practice
        np.random.seed(seed=self.random_seed)

        self.model_data_pre_clustering = self.model_data.copy(deep=True)
        if self.time_config is not None:
            self.model_data = time.apply_time_clustering(
                self.model_data, self.time_config, self.timeseries_data
            )

        self.model_data["annualisation_weight"] = (
            self.model_data.timestep_resolution * self.model_data.timestep_weights
        ).sum() / 8760

    def _clean_model_data(self):
        self._add_var_attrs()
        self._update_dtypes()
        self._check_data()

    @staticmethod
    def _empty_or_invalid(var):
        if isinstance(var, str):
            return False
        elif isinstance(var, list):
            return not var or pd.isnull(var).any()
        elif isinstance(var, (tuple, set, dict)):
            return not var
        else:
            return pd.isnull(var)

    def _strip_unwanted_keys(self):
        """
        These are keys in `model_run` that we don't need in `model_data`.
        Removing them now ensures they don't end up in model_data by mistake
        and that the final `tech_dict` and `node_dicts` should be empty if all
        relevant data *has* made it through to model_data.
        """
        self.stripped_keys = list(
            self._reformat_model_run_dict(
                self.tech_dict,
                [],
                get_method="pop",
                end="({})".format("|".join(self.UNWANTED_TECH_KEYS)),
            ).keys()
        )

        for subdict in ["tech", "node"]:
            for key, val in list(getattr(self, f"{subdict}_dict").items()):
                if self._empty_or_invalid(val):
                    self.stripped_keys.append(key)
                    getattr(self, f"{subdict}_dict").pop(key)

    def _add_node_tech_sets(self):
        """
        Run through the whole `model_run` and extract all the valid combinations of techs
        at nodes.
        """
        kwargs = {"get_method": "get", "end": "\\.({0}).*"}
        df = self._dict_to_df(
            data_dict=self._reformat_model_run_dict(
                self.node_dict, ["techs"], **kwargs
            ),
            data_dimensions=["nodes", "techs"],
            is_link=False,
            var_name="node_tech",
        )

        link_data_dict = self._reformat_model_run_dict(
            self.node_dict, ["links", "techs"], **kwargs
        )
        if not link_data_dict:
            self.link_techs = pd.Series([None])
        else:
            df_link = self._dict_to_df(
                link_data_dict,
                data_dimensions=["nodes", "node_to", "techs"],
                is_link=True,
                var_name="node_tech",
            )
            self._get_link_remotes(link_data_dict)
            df = pd.concat([df, df_link])

        df = self._all_df_to_true(df)

        self.model_data = self.model_data.merge(xr.Dataset.from_dataframe(df))

    def _get_link_remotes(self, link_data_dict):
        df = self._dict_to_df(
            data_dict=link_data_dict,
            data_dimensions=["nodes", "node_to", "techs"],
            is_link=False,
            var_name="node_tech",
        )
        df = df.assign(
            link_remote_techs=df.index.get_level_values("techs")
            + ":"
            + df.index.get_level_values("nodes"),
            link_remote_nodes=df.index.get_level_values("node_to"),
            base_techs=df.index.get_level_values("techs"),
        )
        df_all_link_techs = self._update_link_idx_levels(df)
        self.link_techs = df_all_link_techs["base_techs"].groupby(level="techs").first()

        self.model_data = self.model_data.merge(
            xr.Dataset.from_dataframe(
                df_all_link_techs.drop(["node_tech", "base_techs"], axis=1)
            )
        )

    def _format_lookup(self, string_to_format):
        return string_to_format.format(self.LOOKUP_STR)

    def _get_key_matching_nesting(
        self, nesting, key_to_check, start="({0})\\.", end="\\.({0})", **kwargs
    ):
        nesting_string = "\\.({0})\\.".join(nesting)
        search_string = self._format_lookup(f"^{start}{nesting_string}{end}$")
        return re.search(search_string, key_to_check)

    def _reformat_model_run_dict(
        self,
        model_run_subdict,
        expected_nesting,
        get_method="pop",
        values_as_dimension=False,
        **kwargs,
    ):
        """
        Extract key:value pairs from `model_run` which match the expected dictionary
        nesting. If value is a list, return it as a `.` concatenated string.
        """
        data_dict = {}

        for key in list(model_run_subdict.keys()):
            key_match = self._get_key_matching_nesting(expected_nesting, key, **kwargs)
            if key_match is None:
                continue

            groups = [tuple(key_match.groups())]
            val = getattr(model_run_subdict, get_method)(key)
            if self._empty_or_invalid(val):
                continue
            if values_as_dimension:
                # this if/else is for multiple carriers defined under one carrier tier
                if isinstance(val, list):
                    groups = [groups[0] + (v,) for v in val]
                else:
                    groups[0] += (val,)
                val = 1
            if isinstance(val, list):
                val = ".".join(val)
            for group in groups:
                data_dict[group] = val

        if not data_dict:
            return None
        else:
            return data_dict

    def _dict_to_df(
        self,
        data_dict,
        data_dimensions,
        var_name=None,
        var_name_prefix=None,
        is_link=False,
        **kwargs,
    ):
        """
        Take in a dictionary with tuple keys and turn it into a pandas multi-index dataframe.
        Index levels are data dimensions; columns are data variables.
        """
        df = pd.Series(data_dict)
        if len(data_dimensions) < len(df.index.names):
            df = df.unstack(-1)
        elif var_name is not None:
            df = df.to_frame(var_name)
        df = df.rename_axis(index=data_dimensions)

        if "var_name" in data_dimensions:
            df = df.unstack(data_dimensions.index("var_name"))

        if var_name_prefix is not None:
            df = df.rename(columns=lambda x: var_name_prefix + "_" + x)

        if is_link:
            df = self._update_link_idx_levels(df)

        return df

    def _model_run_dict_to_dataset(
        self,
        group_name,
        model_run_subdict_name,
        expected_nesting,
        data_dimensions,
        **kwargs,
    ):
        """
        Pop out key:value pairs from `model_run` nodes or techs subdicts,
        and turn them into beautifully tabulated, multi-dimensional data in an
        xarray dataset.
        """
        model_run_subdict = getattr(self, f"{model_run_subdict_name}_dict")
        data_dict = self._reformat_model_run_dict(
            model_run_subdict, expected_nesting, **kwargs
        )
        if data_dict is None:
            LOGGER.debug(
                f"Model build | No relevant data found for `{group_name}` group of parameters"
            )
            return None
        df = self._dict_to_df(data_dict, data_dimensions, **kwargs)
        if model_run_subdict_name == "tech":
            df = self._update_link_tech_names(df)
        new_model_data_vars = xr.Dataset.from_dataframe(df)

        self.model_data = self.model_data.combine_first(new_model_data_vars)

    def _update_link_tech_names(self, df):
        """
        tech-specific information will only have info on link techs by their base name,
        but the data needs to be duplicated across all link techs, i.e. for every node
        that tech is linking: (`tech_name` -> [`tech_name:node1`, `tech_name:node2`, ...])
        """
        if isinstance(df.index, pd.MultiIndex):
            idx_to_stack = df.index.names.difference(["techs"])
            df = df.unstack(idx_to_stack)
        else:
            idx_to_stack = []
        if df.index.intersection(self.link_techs.values).empty:
            return df.stack(idx_to_stack)
        df_link_tech_data = df.reindex(self.link_techs.values)
        df_link_tech_data.index = self.link_techs.index
        df = (
            pd.concat([df, df_link_tech_data])
            .drop(self.link_techs.unique(), errors="ignore")
            .dropna(how="all")
            .stack(idx_to_stack)
        )

        return df

    def _add_var_attrs(self):
        for var_data in self.model_data.data_vars.values():
            var_data.attrs["is_result"] = 0

    @staticmethod
    def _update_link_idx_levels(df):
        """
        ([(`tech_name`, `node1`), (`tech_name`, `node2`)] -> [`tech_name:node1`, `tech_name:node2`])
        """
        new_tech = (
            df.index.get_level_values("techs")
            + ":"
            + df.index.get_level_values("node_to")
        )
        df = (
            df.assign(techs=new_tech)
            .droplevel(["techs", "node_to"])
            .set_index("techs", append=True)
        )
        return df

    @staticmethod
    def _all_df_to_true(df):
        return df == df

    def _add_attributes(self, model_run):
        attr_dict = AttrDict()

        attr_dict["calliope_version"] = __version__
        attr_dict["applied_overrides"] = model_run["applied_overrides"]
        attr_dict["scenario"] = model_run["scenario"]

        self.model_data.attrs = attr_dict

    def _add_definition_matrix(self):
        self.model_data["definition_matrix"] = (
            self.model_data.node_tech.notnull() & self.model_data.carrier.notnull()
        )
        self.model_data = self.model_data.drop_vars(["node_tech", "carrier"])

    def _clean_unused_techs_nodes_and_carriers(self):
        """
        Remove techs not assigned to nodes, nodes with no associated techs, and carriers associated with removed techs
        """
        self.model_data["definition_matrix"] = self.model_data.definition_matrix.where(
            self.model_data.definition_matrix
        )
        for dim in ["nodes", "techs", "carriers", "carrier_tiers"]:
            self.model_data = self.model_data.dropna(
                dim, how="all", subset=["definition_matrix"]
            )
        self.model_data[
            "definition_matrix"
        ] = self.model_data.definition_matrix.notnull()
        self.model_data = self.model_data.drop_vars(
            [
                var_name
                for var_name, var in self.model_data.data_vars.items()
                if var.isnull().all()
            ]
        )

    def _add_param_from_template(self):
        for group, group_config in self.template_config.items():
            self._model_run_dict_to_dataset(group_name=group, **group_config)

    def _update_dtypes(self):
        """
        Update dtypes to not be 'Object', if possible.
        Order of preference is: bool, int, float
        """
        # TODO: this should be redundant once typedconfig is in (params will have predefined dtypes)
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

    def _check_data(self):
        if self.node_dict or self.tech_dict:
            raise exceptions.ModelError(
                "Some data not extracted from inputs into model dataset:\n"
                f"{self.node_dict}"
            )
        self.model_data, final_check_comments, warns, errors = checks.check_model_data(
            self.model_data
        )
        exceptions.print_warnings_and_raise_errors(warnings=warns, errors=errors)
