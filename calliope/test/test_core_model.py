import os

import pytest
import tempfile
import pandas as pd
import numpy as np

import calliope
from calliope.test.common.util import check_error_or_warning
from calliope.test.common.util import build_test_model as build_model


class TestModel:
    @pytest.fixture(scope="module")
    def national_scale_example(self):
        model = calliope.examples.national_scale(
            override_dict={"model.subset_time": ["2005-01-01", "2005-01-01"]}
        )
        model.run()
        return model

    @pytest.fixture(params=[dict, calliope.AttrDict])
    def dict_to_add(self, request):
        return request.param({"a": {"b": 1}})

    def test_info(self, national_scale_example):
        national_scale_example.info()

    def test_info_minimal_model(self):
        this_path = os.path.dirname(__file__)
        model_location = os.path.join(
            this_path, "common", "test_model", "model_minimal.yaml"
        )
        model = calliope.Model(model_location)

        model.info()

    def test_update_observed_dict(self, national_scale_example):
        national_scale_example.model_config.name = "foo"
        assert national_scale_example._model_data.attrs["model_config"].name == "foo"

    def test_add_observed_dict_from_model_data(
        self, national_scale_example, dict_to_add
    ):
        national_scale_example._model_data.attrs["foo"] = dict_to_add
        national_scale_example._add_observed_dict("foo")
        assert national_scale_example.foo == dict_to_add
        assert national_scale_example._model_data.attrs["foo"] == dict_to_add

    def test_add_observed_dict_from_dict(self, national_scale_example, dict_to_add):
        national_scale_example._add_observed_dict("bar", dict_to_add)
        assert national_scale_example.bar == dict_to_add
        assert national_scale_example._model_data.attrs["bar"] == dict_to_add

    def test_add_observed_dict_not_available(self, national_scale_example):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            national_scale_example._add_observed_dict("baz")
        assert check_error_or_warning(
            excinfo,
            "Expected the model property `baz` to be a dictionary attribute of the model dataset",
        )
        assert not hasattr(national_scale_example, "baz")

    def test_add_observed_dict_not_dict(self, national_scale_example):
        with pytest.raises(TypeError) as excinfo:
            national_scale_example._add_observed_dict("baz", "bar")
        assert check_error_or_warning(
            excinfo,
            "Attempted to add dictionary property `baz` to model, but received argument of type `str`",
        )


class TestOptimisationConfigOverrides:
    @pytest.fixture
    def storage_inter_cluster(
        self,
    ):
        return build_model(
            {"model.optimisation_config_overrides": ["storage_inter_cluster"]},
            "simple_supply,two_hours,investment_costs",
        )

    @pytest.fixture
    def temp_path(self, tmpdir_factory):
        return tmpdir_factory.mktemp("config_overrides")

    def test_internal_override(self, storage_inter_cluster):
        assert (
            "storage_intra_max"
            in storage_inter_cluster.component_config["constraints"].keys()
        )

    def test_variable_bound(self, storage_inter_cluster):
        assert (
            storage_inter_cluster.component_config["variables"]["storage"]["bounds"][
                "min"
            ]
            == -np.inf
        )

    @pytest.mark.parametrize(
        ["override", "expected"],
        [
            (["foo"], ["foo"]),
            (["bar", "foo"], ["bar", "foo"]),
            (["foo", "storage_inter_cluster"], ["foo"]),
            (["foo.yaml"], ["foo.yaml"]),
        ],
    )
    def test_allowed_internal_constraint(self, override, expected):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            build_model(
                {"model.optimisation_config_overrides": override},
                "simple_supply,two_hours,investment_costs",
            )
        assert check_error_or_warning(
            excinfo,
            f"Attempted to load a configuration override that does not exist: {expected}",
        )

    def test_internal_override_from_yaml(self, temp_path):
        new_constraint = calliope.AttrDict(
            {
                "constraints": {
                    "constraint_name": {"foreach": [], "where": "", "equation": ""}
                }
            }
        )
        new_constraint.to_yaml(temp_path.join("custom-constraints.yaml"))
        m = build_model(
            {
                "model.optimisation_config_overrides": [
                    temp_path.join("custom-constraints.yaml")
                ]
            },
            "simple_supply,two_hours,investment_costs",
        )
        assert "constraint_name" in m.component_config["constraints"].keys()

    def test_override_existing_internal_constraint(self, temp_path, simple_supply):
        file_path = temp_path.join("custom-constraints.yaml")
        new_constraint = calliope.AttrDict(
            {
                "constraints": {
                    "energy_capacity_per_storage_capacity_min": {"foreach": ["nodes"]}
                }
            }
        )
        new_constraint.to_yaml(file_path)
        m = build_model(
            {"model.optimisation_config_overrides": [file_path]},
            "simple_supply,two_hours,investment_costs",
        )
        base = simple_supply.component_config["constraints"][
            "energy_capacity_per_storage_capacity_min"
        ]
        new = m.component_config["constraints"][
            "energy_capacity_per_storage_capacity_min"
        ]

        for i in base.keys():
            if i == "foreach":
                assert new[i] == ["nodes"]
            else:
                assert base[i] == new[i]

    def test_override_order(self, temp_path, simple_supply):
        file_path_1 = temp_path.join("custom-constraints-1.yaml")
        file_path_2 = temp_path.join("custom-constraints-2.yaml")
        new_constraint_1 = calliope.AttrDict(
            {
                "constraints": {
                    "energy_capacity_per_storage_capacity_min": {"foreach": ["nodes"]}
                }
            }
        )
        new_constraint_1.to_yaml(file_path_1)
        new_constraint_2 = calliope.AttrDict(
            {
                "constraints": {
                    "energy_capacity_per_storage_capacity_min": {"foreach": ["techs"]}
                }
            }
        )
        new_constraint_2.to_yaml(file_path_2)
        m = build_model(
            {"model.optimisation_config_overrides": [file_path_1, file_path_2]},
            "simple_supply,two_hours,investment_costs",
        )

        base = simple_supply.component_config["constraints"][
            "energy_capacity_per_storage_capacity_min"
        ]
        new = m.component_config["constraints"][
            "energy_capacity_per_storage_capacity_min"
        ]

        for i in base.keys():
            if i == "foreach":
                assert new[i] == ["techs"]
            else:
                assert base[i] == new[i]

    def test_override_existing_internal_constraint_merge(
        self, temp_path, simple_supply
    ):
        new_constraint = calliope.AttrDict(
            {"variables": {"storage": {"bounds": {"min": -1}}}}
        )
        file_path = temp_path.join("custom-constraints.yaml")
        new_constraint.to_yaml(file_path)
        m = build_model(
            {
                "model.optimisation_config_overrides": [
                    "storage_inter_cluster",
                    file_path,
                ]
            },
            "simple_supply,two_hours,investment_costs",
        )
        base = simple_supply.component_config["variables"]["storage"]
        new = m.component_config["variables"]["storage"]

        for i in base.keys():
            if i == "bounds":
                assert new[i]["min"] == -1
                assert new[i]["max"] == new[i]["max"]
            else:
                assert base[i] == new[i]
