from itertools import chain, combinations

import pytest
import xarray as xr
import numpy as np
import pandas as pd

import calliope
from calliope.backend.subsets import (
    create_valid_subset,
    _param_exists,
    _inheritance,
    _val_is,
    _get_valid_subset,
    _subset_imask,
    _imask_where,
    _combine_imasks,
    _imask_foreach,
)
from calliope.core.util.observed_dict import UpdateObserverDict
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
        UpdateObserverDict(
            initial_dict=AttrDict({"foo": True, "baz": {"bar": "foobar"}}),
            name="run_config",
            observer=model_data,
        )
        UpdateObserverDict(
            initial_dict={"foz": 0}, name="model_config", observer=model_data
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

    @pytest.mark.parametrize(
        "foreach",
        set(
            chain.from_iterable(
                combinations(BASE_DIMS, i) for i in range(1, len(BASE_DIMS))
            )
        ),
    )
    def test_foreach_constraint(self, model_data, foreach):
        imask = _imask_foreach(model_data, foreach)

        assert sorted(imask.dims) == sorted(foreach)

    @pytest.mark.parametrize(
        ("param", "result"), (("inexistent", False), ("all_inf", 0), ("with_inf", 5))
    )
    def test_param_exists(self, model_data, param, result):
        imask = _param_exists(model_data, param)
        if param == "inexistent":
            assert imask is result
        else:
            assert imask.sum() == result

    @pytest.mark.parametrize(
        ("tech_group", "result"), (("foo", 0), ("bar", 2), ("baz", 1))
    )
    def test_inheritance(self, model_data, tech_group, result):
        imask = _inheritance(model_data, tech_group)
        assert imask.sum() == result

    @pytest.mark.parametrize(
        ("param", "val", "result"),
        (
            ("run.foo", "True", True),
            ("run.foo", "False", False),
            ("run.baz.bar", "'foobar'", True),
            ("run.foobar", "True", False),
            ("model.foz", "False", True),
            ("inexistent", "True", False),
            ("with_inf", "2", 1),
            ("with_inf", "0", 0),
            ("with_inf", "True", 3),
            ("with_inf", "2.0", 1),
        ),
    )
    def test_val_is(self, model_data, param, val, result):
        imask = _val_is(model_data, param, val)
        if isinstance(result, bool):
            assert imask is result
        else:
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

    def test_subset_imask_squeeze(self, model_data, imask_subset_config):
        """
        Include an additional dimension in 'where', which we then subset on (and squeeze out)
        """
        # foreach doesn't have this additional dimension
        foreach = ["techs"]
        imask = _imask_foreach(model_data, foreach)
        assert imask.dims == ("techs",)
        # on using 'where', the 'nodes' dimension is added
        imask = _imask_where(model_data, "foo", ["node_tech"], imask, "and_")
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

    def test_imask_where_ineritance(self, model_data):
        where_imask = _imask_where(model_data, "foo", ["inheritance(bar)"])
        assert (
            where_imask == [[True, False, False, False], [True, False, False, False]]
        ).all()

    @pytest.mark.parametrize("param", ("node_tech", "with_inf", "inexistent"))
    def test_imask_where_param_exists(self, model_data, param):
        where_imask = _imask_where(model_data, "foo", [param])
        if param in model_data:
            assert (where_imask == _param_exists(model_data, param)).all()
        else:
            assert where_imask is False

    @pytest.mark.parametrize(
        ("val", "result"),
        (
            ("run.foo=True", True),
            ("with_inf=3", [[False, False, False, True], [False, False, False, False]]),
            ("inexistent='foo'", False),
        ),
    )
    def test_imask_where_val_is(self, model_data, val, result):
        where_imask = _imask_where(model_data, "foo", [val])
        assertion = where_imask == result
        if isinstance(result, list):
            assertion = assertion.all()
        assert assertion

    def test_imask_where_not(self, model_data):
        where_imask = _imask_where(model_data, "foo", ["with_inf"])
        not_where_imask = _imask_where(model_data, "foo", ["not with_inf"])
        assert where_imask.equals(~not_where_imask)

    @pytest.mark.parametrize(
        ("where_array", "results"),
        (
            (["with_inf", "and", "inheritance(bar)"], (True, False)),
            (["with_inf=4", "or", "inheritance(bar)"], (True, True)),
            (
                [["with_inf", "or", "inheritance(bar)"], "and", "inheritance='bar'"],
                (False, True),
            ),
            (
                [["with_inf", "and", "inheritance(bar)"], "or", "inheritance='bar'"],
                (True, True),
            ),
        ),
    )
    def test_imask_where_array(self, model_data, where_array, results):
        where_imask = _imask_where(model_data, "foo", where_array)
        assert (
            where_imask
            == [[results[0], False, False, False], [results[1], False, False, False]]
        ).all()

    @pytest.mark.parametrize(
        ("where_array", "results"),
        (
            (["with_inf", "and", "inheritance(bar)"], (True, False)),
            (["with_inf=4", "or", "inheritance(bar)"], (True, True)),
            (
                [["with_inf", "or", "inheritance(bar)"], "and", "inheritance='bar'"],
                (False, True),
            ),
            (
                [["with_inf", "and", "inheritance(bar)"], "or", "inheritance='bar'"],
                (True, True),
            ),
        ),
    )
    @pytest.mark.parametrize("initial_operator", ("and_", "or_"))
    def test_imask_where_initial_imask(
        self, model_data, where_array, results, initial_operator
    ):
        foreach = ["nodes", "techs", "carriers"]
        imask = _imask_foreach(model_data, foreach)
        where_imask = _imask_where(
            model_data, "foo", where_array, imask, initial_operator
        )
        locs = [{"nodes": "foo", "techs": "foo"}, {"nodes": "bar", "techs": "foo"}]
        for i in range(2):
            if initial_operator == "and_":
                assert (
                    imask.loc[locs[i]] * results[i] == where_imask.loc[locs[i]]
                ).all()
            elif initial_operator == "or_":
                assert (
                    imask.loc[locs[i]] + results[i] == where_imask.loc[locs[i]]
                ).all()

    def test_imask_where_incorrect_where(self, model_data):
        with pytest.raises(ValueError) as excinfo:
            _imask_where(model_data, "foo", ["node_tech", "inheritance(bar)"])
        assert check_error_or_warning(excinfo, "'where' array for set `foo` must")

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

    @pytest.mark.parametrize(
        ("operator", "result"), (("or", True), ("and", False), ("foo", "error"))
    )
    def test_combine_imask(self, operator, result):
        curr = False
        new = True
        if isinstance(result, bool):
            assert _combine_imasks(curr, new, operator) is result
        elif result == "error":
            with pytest.raises(ValueError) as excinfo:
                _combine_imasks(curr, new, operator)
            assert check_error_or_warning(excinfo, "Operator `foo` not recognised")
