"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying param_dict.

"""
import os
import re
import logging

import xarray as xr
import numpy as np
import pandas as pd

import calliope
from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope._version import __version__
from calliope.preprocess import checks
from calliope.preprocess import time
from calliope.core.util import dataset


def build_model_data(model_run, debug=False):
    """
    Take a Calliope model_run and convert it into an xarray Dataset, ready for
    constraint generation. Timeseries data is also extracted from file at this
    point, and the time dimension added to the data

    ----------
    model_run : AttrDict
        preprocessed model_run dictionary, as produced by
        Calliope.preprocess.preprocess_model
    debug : bool, default = False
        Used to debug steps within build_model_data, particularly before/after
        time dimension addition. If True, more information is returned

    Returns
    -------
    data : xarray Dataset
        Dataset with optimisation param_dict as variables, optimisation sets as
        coordinates, and other information in attributes.
    data_dict : dict, only returned if debug = True
        dictionary of param_dict, prior to time dimension addition. Used here to
        populate the Dataset (using `from_dict()`)
    data_pre_time : xarray Dataset, only returned if debug = True
        Dataset, prior to time dimension addition, with optimisation param_dict
        as variables, optimisation sets as coordinates, and other information
        in attributes.
    """
    # We build up a dictionary of the data, then convert it to an xarray Dataset
    # before applying time dimensions
    model_data = ModelData(model_run)

    model_data.add_param_from_template()
    model_data.clean_unused_techs_nodes_and_carriers()

    if debug:
        data_pre_time = model_data.model_data.copy(deep=True)

    model_data.add_time_dimension(model_run)
    model_data.apply_time_clustering(model_run)
    model_data.reorganise_xarray_dimensions()
    model_data.add_var_attrs()
    model_data.update_dtypes()
    model_data.check_data()

    if debug:
        return model_data.model_data_original, model_data.model_data, data_pre_time
    else:
        return model_data.model_data_original, model_data.model_data


class ModelData:
    """
    Class to generate Calliope xarray model dataset, known as `model_data`.

    Takes as input the calliope `model_run` dictionary, reorganises the data into
    tabular form for inclusion in `model_data`.
    The method for reorganising `model_run` depends on regular expressions,
    configuration for which is found in `calliope/config/model_data_lookup.yaml`.
    """

    def __init__(self, model_run_dict):
        self.node_dict = model_run_dict.nodes.as_dict_flat()
        self.tech_dict = model_run_dict.techs.as_dict_flat()

        self.model_data = xr.Dataset(
            coords={"timesteps": model_run_dict.timeseries_data.index}
        )
        self.add_attributes(model_run_dict)
        self.lookup_str = "[\\w\\-]*"  # all alphanumerics + `_` and `-`
        self.template_config = AttrDict.from_yaml(
            os.path.join(
                os.path.dirname(calliope.__file__), "config", "model_data_lookup.yaml"
            )
        )
        self.unwanted_tech_keys = [
            "allowed_constraints",
            "required_constraints",
            "allowed_group_constraints",
            "allowed_costs",
            "allowed_switches",
            "constraints",
            "essentials.carrier",
        ]

        self.strip_unwanted_keys()
        self.add_node_tech_sets()

        if model_run_dict.get_key("model.random_seed", None):
            np.random.seed(seed=model_run_dict.model.random_seed)

    def empty_or_invalid(self, var):
        if isinstance(var, str):
            return False
        elif isinstance(var, (tuple, set, list)):
            return not var or pd.isnull(var).any()
        elif isinstance(var, dict):
            return not var
        else:
            return pd.isnull(var)

    def strip_unwanted_keys(self):
        """
        These are keys in `model_run` that we don't need in `model_data`.
        Removing them now ensures they don't end up in model_data by mistake
        and that the final `tech_dict` and `node_dicts` should be empty if all
        relevant data *has* made it through to model_data.
        """
        self.stripped_keys = list(
            self.reformat_model_run_dict(
                self.tech_dict,
                [],
                get_method="pop",
                end="({})".format("|".join(self.unwanted_tech_keys)),
            ).keys()
        )

        for subdict in ["tech", "node"]:
            for key, val in list(getattr(self, f"{subdict}_dict").items()):
                if self.empty_or_invalid(val):
                    self.stripped_keys.append(key)
                    getattr(self, f"{subdict}_dict").pop(key)

    def add_node_tech_sets(self):
        """
        Run through the whole `model_run` and extract all the valid combinations of techs
        at nodes.
        """
        kwargs = {"get_method": "get", "end": "\\.({0}).*"}
        df = self.dict_to_df(
            data_dict=self.reformat_model_run_dict(self.node_dict, ["techs"], **kwargs),
            data_dimensions=["nodes", "techs"],
            is_link=False,
            var_name="node_tech",
        )

        link_data_dict=self.reformat_model_run_dict(
            self.node_dict, ["links", "techs"], **kwargs
        )
        if not link_data_dict:
            self.link_techs = pd.DataFrame([None])
        else:
            df_link = self.dict_to_df(link_data_dict,
                data_dimensions=["nodes", "node_to", "techs"],
                is_link=True,
                var_name="node_tech",
            )
            self.get_link_remotes(link_data_dict)
            df = df.append(df_link)

        df = self.all_df_to_true(df)

        self.model_data = self.model_data.merge(xr.Dataset.from_dataframe(df))

    def get_link_remotes(self, link_data_dict):
        df = self.dict_to_df(
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
        df_all_link_techs = self.update_link_idx_levels(df)
        self.link_techs = df_all_link_techs["base_techs"].groupby(level="techs").first()

        self.model_data = self.model_data.merge(
            xr.Dataset.from_dataframe(
                df_all_link_techs.drop(["node_tech", "base_techs"], axis=1)
            )
        )

    def format_lookup(self, string_to_format):
        return string_to_format.format(self.lookup_str)

    def get_key_matching_nesting(
        self, nesting, key_to_check, start="({0})\\.", end="\\.({0})", **kwargs
    ):
        nesting_string = "\\.({0})\\.".join(nesting)
        search_string = self.format_lookup(f"^{start}{nesting_string}{end}$")
        return re.search(search_string, key_to_check)

    def reformat_model_run_dict(
        self, model_run_subdict, expected_nesting, get_method="pop",
        values_as_dimension=False, **kwargs
    ):
        """
        Extract key:value pairs from `model_run` which match the expected dictionary
        nesting. If value is a list, return it as a `.` concatenated string.
        """
        data_dict = {}

        for key in list(model_run_subdict.keys()):
            key_match = self.get_key_matching_nesting(expected_nesting, key, **kwargs)
            if key_match is None:
                continue

            groups = [tuple(key_match.groups())]
            val = getattr(model_run_subdict, get_method)(key)
            if self.empty_or_invalid(val):
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

    def dict_to_df(
        self,
        data_dict,
        data_dimensions,
        var_name=None,
        var_name_prefix=None,
        is_link=False,
        **kwargs
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
            df = self.update_link_idx_levels(df)

        return df

    def model_run_dict_to_dataset(
        self,
        group_name,
        model_run_subdict_name,
        expected_nesting,
        data_dimensions,
        **kwargs
    ):
        """
        Pop out key:value pairs from `model_run` nodes or techs subdicts,
        and turn them into beautifully tabulated, multi-dimensional data in an
        xarray dataset.
        """
        model_run_subdict = getattr(self, f"{model_run_subdict_name}_dict")
        data_dict = self.reformat_model_run_dict(
            model_run_subdict, expected_nesting, **kwargs
        )
        if not data_dict:
            logging.info(
                f"No relevant data found for `{group_name}` group of parameters"
            )
            return None
        df = self.dict_to_df(data_dict, data_dimensions, **kwargs)
        if model_run_subdict_name == "tech":
            df = self.update_link_tech_names(df)
        new_model_data_vars = xr.Dataset.from_dataframe(df)

        self.model_data = self.model_data.combine_first(new_model_data_vars)

    def update_link_tech_names(self, df):
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
            df.append(df_link_tech_data)
            .drop(self.link_techs.unique(), errors="ignore")
            .dropna(how='all')
            .stack(idx_to_stack)
        )

        return df

    def add_var_attrs(self):
        for var_data in self.model_data.data_vars.values():
            var_data.attrs["parameters"] = 1
            var_data.attrs["is_result"] = 0

    def update_link_idx_levels(self, df):
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

    def all_df_to_true(self, df):
        return df == df

    def add_attributes(self, model_run):
        attr_dict = AttrDict()

        attr_dict["calliope_version"] = __version__
        attr_dict["applied_overrides"] = model_run["applied_overrides"]
        attr_dict["scenario"] = model_run["scenario"]

        default_tech_dict = checks.DEFAULTS.techs.default_tech
        default_cost_dict = {
            "cost_{}".format(k): v
            for k, v in default_tech_dict.costs.default_cost.items()
        }
        default_node_dict = checks.DEFAULTS.nodes.default_node

        attr_dict["defaults"] = AttrDict(
            {
                **default_tech_dict.constraints.as_dict(),
                **default_tech_dict.switches.as_dict(),
                **default_cost_dict,
                **default_node_dict.as_dict(),
            }
        ).to_yaml()

        self.model_data.attrs = attr_dict

    def clean_unused_techs_nodes_and_carriers(self):
        """
        Remove techs not assigned to nodes, nodes with no associated techs, and carriers associated with removed techs
        """
        for dim in ["nodes", "techs"]:
            self.model_data = self.model_data.dropna(
                dim, how="all", subset=["node_tech"]
            )
        for dim in ["carriers", "carrier_tiers"]:
            self.model_data = self.model_data.dropna(dim, how="all")

        self.model_data = self.model_data.drop_vars(
            [
                var_name
                for var_name, var in self.model_data.data_vars.items()
                if var.isnull().all()
            ]
        )

    def add_param_from_template(self):
        for group, group_config in self.template_config.items():
            self.model_run_dict_to_dataset(group_name=group, **group_config)

    def add_time_dimension(self, model_run):
        self.model_data = time.add_time_dimension(self.model_data, model_run)
        self.update_dtypes()
        self.model_data = time.add_max_demand_timesteps(self.model_data)

    def apply_time_clustering(self, model_run):
        self.model_data_original = self.model_data.copy(deep=True)
        time_config = model_run.model.get("time", None)
        if time_config:
            self.model_data = time.apply_time_clustering(self.model_data, model_run)

    def reorganise_xarray_dimensions(self):
        self.model_data = dataset.reorganise_xarray_dimensions(self.model_data)

    def update_dtypes(self):
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
                        self.model_data[var_name] = var.astype(np.int, copy=False)
                    except (ValueError, OverflowError):
                        try:
                            self.model_data[var_name] = var.astype(np.float, copy=False)
                        except ValueError:
                            None

    def check_data(self):
        if self.node_dict or self.tech_dict:
            raise exceptions.ModelError(
                "Some data not extracted from inputs into model dataset:\n"
                f"{self.node_dict}"
            )
        self.model_data, final_check_comments, warns, errors = checks.check_model_data(
            self.model_data
        )
        exceptions.print_warnings_and_raise_errors(warnings=warns, errors=errors)
