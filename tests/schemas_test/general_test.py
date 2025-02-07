import logging

import numpy as np
import pydantic
import pytest

from calliope.schemas import general


class TestUniqueList:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model("Model", unique_list=(general.UniqueList, ...))

    @pytest.mark.parametrize(
        "valid_list",
        [[1, 2, 3], [1.0, 1.1, 1.2], ["1", "2", "3"], ["1", 1, "foo"], [None, np.nan]],
    )
    def test_unique_list(self, pydantic_model, valid_list):
        "When there's no fixed type for list entries, they just have to be unique _within_ types"
        pydantic_model = pydantic_model(unique_list=valid_list)
        assert pydantic_model.unique_list == valid_list

    @pytest.mark.parametrize("valid_list", [[1, 2, 3], ["1", "2", "3"], ["foo", "bar"]])
    def test_unique_str_list(self, pydantic_model, valid_list):
        "When there's a fixed type for list entries, they have to be unique when coerced to that type"
        pydantic_model = pydantic_model(unique_list=valid_list)
        assert pydantic_model.unique_list == valid_list

    @pytest.mark.parametrize(
        "invalid_list",
        [[1, 1, 2], [1, 1.0], ["1", "foo", "foo"], [None, None], [1, True], [0, False]],
    )
    def test_not_unique_list(self, pydantic_model, invalid_list):
        "When there's no fixed type for list entries, duplicate entries of the _same_ type is not allowed (includes int == bool)"
        with pytest.raises(pydantic.ValidationError, match="List must be unique"):
            pydantic_model(unique_list=invalid_list)

    @pytest.mark.parametrize(
        "invalid_list",
        [
            [1, 1, 2],
            [1, 1.0, 2, "2"],
            ["foo", 1, "foo"],
            ["1", "foo", "foo"],
            [[1, 1, 2], [1, 1, 2], ["1", "foo", 3]],
        ],
    )
    def test_not_unique_str_list(self, pydantic_model, invalid_list):
        "When there's a fixed type for list entries, they have to be unique when coerced to that type."
        with pytest.raises(pydantic.ValidationError, match="List must be unique"):
            pydantic_model(unique_list=invalid_list)


class TestNonEmptyList:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model(
            "Model", non_empty_list=(general.NonEmptyList, ...)
        )

    def test_invalid_input(self, pydantic_model):
        """Passing empty lists should result in a validation error."""
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(non_empty_list=[])


class TestNonEmptyUniqueList:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model(
            "Model", non_empty_unique_list=(general.NonEmptyUniqueList, ...)
        )

    @pytest.mark.parametrize(
        "invalid_input", [[], [1, 1], ["1", "foo", "foo"], [[1, 2], [1, 2], [1]]]
    )
    def test_invalid_input(self, pydantic_model, invalid_input):
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(non_empty_unique_list=invalid_input)


class TestAttrStr:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model("Model", attrstr=(general.AttrStr, ...))

    @pytest.mark.parametrize(
        "invalid_input", [1, "1thing", "with spaces", "111992", None, True]
    )
    def test_invalid_input(self, pydantic_model, invalid_input):
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(attrstr=invalid_input)


class TestNumveriVal:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model("Model", numeric_val=(general.NumericVal, ...))

    @pytest.mark.parametrize("invalid_input", [[1, 2], "foobar", None])
    def test_invalid_input(self, pydantic_model, invalid_input):
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(numeric_val=invalid_input)


class TestCalliopeBaseModel:
    @pytest.fixture(scope="module")
    def config_model_flat(self):
        return pydantic.create_model(
            "Model",
            __base__=general.CalliopeBaseModel,
            model_config={"title": "TITLE"},
            foo=(str, "bar"),
            foobar=(int, 1),
        )

    @pytest.fixture(scope="module")
    def config_model_nested(self, config_model_flat):
        return pydantic.create_model(
            "Model",
            __base__=general.CalliopeBaseModel,
            model_config={"title": "TITLE 2"},
            nested=(config_model_flat, config_model_flat()),
            top_level_foobar=(int, 10),
        )

    @pytest.fixture(scope="module")
    def config_model_double_nested(self, config_model_nested):
        return pydantic.create_model(
            "Model",
            __base__=general.CalliopeBaseModel,
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

        with pytest.raises(
            pydantic.ValidationError, match="1 validation error for TITLE"
        ):
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

    @pytest.fixture(scope="module")
    def config_model_submodel(self):
        sub_model = pydantic.create_model(
            "SubModel",
            __base__=general.CalliopeBaseModel,
            model_config={"title": "TITLE"},
            foo=(str, "bar"),
            foobar=(int, 1),
        )
        model = pydantic.create_model(
            "Model",
            __base__=general.CalliopeBaseModel,
            model_config={"title": "TITLE 2"},
            nested=(sub_model, sub_model()),
        )
        return model

    def test_config_model_no_defs(self, config_model_submodel):
        model = config_model_submodel()
        json_schema = model.model_json_schema()
        no_defs_json_schema = model.model_no_ref_schema()
        assert "$defs" in json_schema
        assert "$defs" not in no_defs_json_schema

    def test_config_model_no_resolved_refs(self, config_model_submodel):
        model = config_model_submodel()
        json_schema = model.model_json_schema()
        no_defs_json_schema = model.model_no_ref_schema()
        assert json_schema["properties"]["nested"] == {
            "$ref": "#/$defs/SubModel",
            "default": {"foo": "bar", "foobar": 1},
        }
        assert (
            no_defs_json_schema["properties"]["nested"]
            == json_schema["$defs"]["SubModel"]
        )
