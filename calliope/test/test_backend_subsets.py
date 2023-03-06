from itertools import chain, combinations

import pytest
import xarray as xr
import numpy as np
import pandas as pd

import calliope
from calliope.backend.subsets import (
    create_valid_subset,
    _inheritance,
    _get_valid_subset,
    _subset_imask,
    _imask_foreach,
    VALID_HELPER_FUNCTIONS,
)
from calliope.backend.subset_parser import generate_where_string_parser

from calliope import AttrDict
from calliope.test.common.util import (
    check_error_or_warning,
    constraint_sets,
    subsets_config,
)

BASE_DIMS = ["nodes", "techs", "carriers", "costs", "timesteps", "carrier_tiers"]


class TestSubsets:
    def parse_yaml(self, yaml_string):
        return AttrDict.from_yaml_string(yaml_string)

    @pytest.fixture
    def model_data(self):
        model_data = xr.Dataset(
            coords={
                dim: ["foo", "bar"]
                if dim != "techs"
                else ["foo", "bar", "foobar", "foobaz"]
                for dim in BASE_DIMS
            },
            data_vars={
                "node_tech": (
                    ["nodes", "techs"],
                    np.random.choice(a=[np.nan, True], size=(2, 4)),
                ),
                "carrier": (
                    ["carrier_tiers", "carriers", "techs"],
                    np.random.choice(a=[np.nan, True], size=(2, 2, 4)),
                ),
                "with_inf": (
                    ["nodes", "techs"],
                    [[1.0, np.nan, 1.0, 3], [np.inf, 2.0, True, np.nan]],
                ),
                "all_inf": (["nodes", "techs"], np.ones((2, 4)) * np.inf),
                "all_nan": (["nodes", "techs"], np.ones((2, 4)) * np.nan),
                "inheritance": (
                    ["nodes", "techs"],
                    [
                        ["foo.bar", "boo", "baz", "boo"],
                        ["bar", "ar", "baz.boo", "foo.boo"],
                    ],
                ),
            },
        )
        model_data.attrs["run_config"] = AttrDict(
            {"foo": True, "baz": {"bar": "foobar"}}
        )
        model_data.attrs["model_config"] = AttrDict({"foz": 0})
        model_data.attrs["defaults"] = AttrDict(
            {"all_inf": np.inf, "all_nan": np.nan, "with_inf": 100}
        )

        return model_data

    @pytest.fixture
    def imask_subset_config(self):
        def _imask_subset_config(foreach):
            return self.parse_yaml(
                f"""
                    foreach: {foreach}
                    subset.nodes: [foo]
                """
            )

        return _imask_subset_config

    @pytest.fixture
    def evaluated_imask_where(self, model_data):
        def _evaluated_imask_where(where_string):
            return (
                generate_where_string_parser()
                .parse_string(where_string, parse_all=True)[0]
                .eval(
                    model_data=model_data,
                    helper_func_dict=VALID_HELPER_FUNCTIONS,
                    defaults=model_data.attrs["defaults"],
                )
            )

        return _evaluated_imask_where

    @pytest.mark.parametrize(
        ("tech_group", "result"), (("foo", 0), ("bar", 2), ("baz", 1))
    )
    def test_inheritance(self, model_data, tech_group, result):
        imask = _inheritance(model_data)(tech_group)
        assert imask.sum() == result

    @pytest.mark.parametrize(
        "foreach", (["techs"], ["nodes", "techs"], ["nodes", "techs", "carriers"])
    )
    def test_get_valid_subset(self, model_data, foreach):
        imask = _imask_foreach(model_data, foreach)
        idx = _get_valid_subset(imask)
        assert isinstance(idx, pd.Index)
        assert len(idx) == imask.sum()
        assert all(imask.loc[i] == 1 for i in idx)  # 1 represents boolean True here

    def test_subset_imask_no_squeeze(self, model_data, imask_subset_config):
        """
        Subset on nodes
        """
        foreach = ["nodes", "techs"]
        imask = _imask_foreach(model_data, foreach)
        assert set(imask.dims) == set(foreach)
        imask_subset = _subset_imask("foo", imask_subset_config(foreach), imask)
        assert (
            imask_subset.loc[{"nodes": "bar"}] == 0
        ).all()  # 0 represents boolean False here

    def test_subset_imask_squeeze(
        self, model_data, imask_subset_config, evaluated_imask_where
    ):
        """
        Include an additional dimension in 'where', which we then subset on (and squeeze out)
        """
        # foreach doesn't have this additional dimension
        foreach = ["techs"]
        imask = _imask_foreach(model_data, foreach)
        assert imask.dims == ("techs",)
        # on using 'where', the 'nodes' dimension is added
        imask = evaluated_imask_where("with_inf")
        assert sorted(imask.dims) == sorted(["nodes", "techs"])
        imask_subset = _subset_imask("foo", imask_subset_config(foreach), imask)
        assert imask_subset.dims == ("techs",)
        assert imask_subset.equals(imask.loc[{"nodes": "foo"}].drop_vars("nodes"))

    def test_subset_imask_non_iterable_subset(self, model_data, imask_subset_config):
        """
        Subset using a string, when a non-string iterable is required
        """
        foreach = ["nodes", "techs"]
        imask = _imask_foreach(model_data, foreach)

        with pytest.raises(TypeError) as excinfo:
            _subset_imask(
                "foo", AttrDict({"foreach": foreach, "subset.nodes": "bar"}), imask
            )
        assert check_error_or_warning(
            excinfo,
            "set `foo` must subset over an iterable, instead got non-iterable `bar` for subset `nodes`",
        )

    @pytest.mark.parametrize("model_name", ("urban_scale", "national_scale", "milp"))
    def test_create_valid_subset(self, model_name):
        model = getattr(calliope.examples, model_name)()

        for object_type in ["constraints", "expressions"]:
            valid_subsets = {
                name: create_valid_subset(model._model_data, name, config)
                for name, config in subsets_config[object_type].items()
            }

            for name, subset in valid_subsets.items():
                if subset is None:
                    continue
                if "timesteps" in subset.names:
                    subset = subset.droplevel("timesteps").unique()
                # FIXME: simplified comparison since constraint_sets.yaml isn't completely cleaned
                # up to match current representation of set elements
                assert len(
                    constraint_sets[f"{model_name}.{object_type}.{name}"]
                ) == len(subset)
