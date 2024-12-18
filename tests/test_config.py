import logging

import numpy as np
import pydantic
import pytest
from pydantic_core import ValidationError

from calliope import config


class TestUniqueList:
    @pytest.fixture(scope="module")
    def unique_list_model(self):
        return pydantic.create_model("Model", unique_list=(config.UniqueList, ...))

    @pytest.fixture(scope="module")
    def unique_str_list_model(self):
        return pydantic.create_model("Model", unique_list=(config.UniqueList[str], ...))

    @pytest.mark.parametrize(
        "valid_list",
        [[1, 2, 3], [1.0, 1.1, 1.2], ["1", "2", "3"], ["1", 1, "foo"], [None, np.nan]],
    )
    def test_unique_list(self, unique_list_model, valid_list):
        "When there's no fixed type for list entries, they just have to be unique _within_ types"
        model = unique_list_model(unique_list=valid_list)
        assert model.unique_list == valid_list

    @pytest.mark.parametrize("valid_list", [[1, 2, 3], ["1", "2", "3"], ["foo", "bar"]])
    def test_unique_str_list(self, unique_list_model, valid_list):
        "When there's a fixed type for list entries, they have to be unique when coerced to that type"
        model = unique_list_model(unique_list=valid_list)
        assert model.unique_list == valid_list

    @pytest.mark.parametrize(
        "invalid_list",
        [[1, 1, 2], [1, 1.0], ["1", "foo", "foo"], [None, None], [1, True], [0, False]],
    )
    def test_not_unique_list(self, unique_list_model, invalid_list):
        "When there's no fixed type for list entries, duplicate entries of the _same_ type is not allowed (includes int == bool)"
        with pytest.raises(ValidationError, match="List must be unique"):
            unique_list_model(unique_list=invalid_list)

    @pytest.mark.parametrize(
        "invalid_list", [[1, 1, 2], ["foo", 1, "foo"], ["1", "foo", "foo"]]
    )
    def test_not_unique_str_list(self, unique_list_model, invalid_list):
        "When there's a fixed type for list entries, they have to be unique when coerced to that type"
        with pytest.raises(ValidationError, match="List must be unique"):
            unique_list_model(unique_list=invalid_list)


class TestUpdate:
    @pytest.fixture(scope="module")
    def config_model_flat(self):
        return pydantic.create_model(
            "Model",
            __base__=config.ConfigBaseModel,
            model_config={"title": "TITLE"},
            foo=(str, "bar"),
            foobar=(int, 1),
        )

    @pytest.fixture(scope="module")
    def config_model_nested(self, config_model_flat):
        return pydantic.create_model(
            "Model",
            __base__=config.ConfigBaseModel,
            model_config={"title": "TITLE 2"},
            nested=(config_model_flat, config_model_flat()),
            top_level_foobar=(int, 10),
        )

    @pytest.fixture(scope="module")
    def config_model_double_nested(self, config_model_nested):
        return pydantic.create_model(
            "Model",
            __base__=config.ConfigBaseModel,
            model_config={"title": "TITLE 3"},
            extra_nested=(config_model_nested, config_model_nested()),
        )

    @pytest.mark.parametrize(
        ("to_update", "expected"),
        [
            ({"foo": "baz"}, {"foo": "baz", "foobar": 1}),
            ({"foobar": 2}, {"foo": "bar", "foobar": 2}),
            ({"foo": "baz", "foobar": 2}, {"foo": "baz", "foobar": 2}),
        ],
    )
    def test_update_flat(self, config_model_flat, to_update, expected):
        model = config_model_flat()
        model_dict = model.model_dump()

        new_model = model.update(to_update)

        assert new_model.model_dump() == expected
        assert model.model_dump() == model_dict

    @pytest.mark.parametrize(
        ("to_update", "expected"),
        [
            (
                {"top_level_foobar": 20},
                {"top_level_foobar": 20, "nested": {"foo": "bar", "foobar": 1}},
            ),
            (
                {"nested": {"foobar": 2}},
                {"top_level_foobar": 10, "nested": {"foo": "bar", "foobar": 2}},
            ),
            (
                {"top_level_foobar": 20, "nested": {"foobar": 2}},
                {"top_level_foobar": 20, "nested": {"foo": "bar", "foobar": 2}},
            ),
            (
                {"top_level_foobar": 20, "nested.foobar": 2},
                {"top_level_foobar": 20, "nested": {"foo": "bar", "foobar": 2}},
            ),
        ],
    )
    def test_update_nested(self, config_model_nested, to_update, expected):
        model = config_model_nested()
        model_dict = model.model_dump()

        new_model = model.update(to_update)

        assert new_model.model_dump() == expected
        assert model.model_dump() == model_dict

    @pytest.mark.parametrize(
        "to_update",
        [
            {"extra_nested.nested.foobar": 2},
            {"extra_nested": {"nested": {"foobar": 2}}},
        ],
    )
    def test_update_extra_nested(self, config_model_double_nested, to_update):
        model = config_model_double_nested()
        model_dict = model.model_dump()

        new_model = model.update(to_update)

        assert new_model.extra_nested.nested.foobar == 2
        assert model.model_dump() == model_dict

    @pytest.mark.parametrize(
        "to_update",
        [
            {"extra_nested.nested.foobar": "foo"},
            {"extra_nested.top_level_foobar": "foo"},
        ],
    )
    def test_update_extra_nested_validation_error(
        self, config_model_double_nested, to_update
    ):
        model = config_model_double_nested()

        with pytest.raises(ValidationError, match="1 validation error for TITLE"):
            model.update(to_update)

    @pytest.mark.parametrize(
        ("to_update", "expected"),
        [
            ({"extra_nested.nested.foobar": 2}, ["Updating TITLE `foobar`: 1 -> 2"]),
            (
                {"extra_nested.top_level_foobar": 2},
                ["Updating TITLE 2 `top_level_foobar`: 10 -> 2"],
            ),
            (
                {"extra_nested.nested.foobar": 2, "extra_nested.top_level_foobar": 3},
                [
                    "Updating TITLE `foobar`: 1 -> 2",
                    "Updating TITLE 2 `top_level_foobar`: 10 -> 3",
                ],
            ),
        ],
    )
    def test_logging(self, caplog, config_model_double_nested, to_update, expected):
        caplog.set_level(logging.INFO)

        model = config_model_double_nested()
        model.update(to_update)

        assert all(log_text in caplog.text for log_text in expected)
