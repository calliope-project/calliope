import logging

import numpy as np
import pydantic
import pytest

from calliope.schemas import general

LOGGER = "calliope.schemas.general"


class TestUniqueList:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model("TestModel", unique_list=(general.UniqueList, ...))

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
            "TestModel", non_empty_list=(general.NonEmptyList, ...)
        )

    def test_invalid_input(self, pydantic_model):
        """Passing empty lists should result in a validation error."""
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(non_empty_list=[])


class TestNonEmptyUniqueList:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model(
            "TestModel", non_empty_unique_list=(general.NonEmptyUniqueList, ...)
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
        return pydantic.create_model("TestModel", attrstr=(general.AttrStr, ...))

    @pytest.mark.parametrize(
        "invalid_input", [1, "1thing", "with spaces", "111992", None, True]
    )
    def test_invalid_input(self, pydantic_model, invalid_input):
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(attrstr=invalid_input)


class TestNumveriVal:
    @pytest.fixture(scope="class")
    def pydantic_model(self):
        return pydantic.create_model("TestModel", numeric_val=(general.NumericVal, ...))

    @pytest.mark.parametrize("invalid_input", [[1, 2], "foobar", None])
    def test_invalid_input(self, pydantic_model, invalid_input):
        with pytest.raises(pydantic.ValidationError):
            pydantic_model(numeric_val=invalid_input)


class TestCalliopeDictModel:
    @pytest.fixture(scope="class")
    def dict_str_model(self):
        dict_model = pydantic.create_model(
            "TestModel", __base__=general.CalliopeDictModel, root=(dict[str, str], {})
        )
        return dict_model({"key1": "value1", "key2": "value2"})

    @pytest.fixture(scope="class")
    def base_model(self):
        return pydantic.create_model(
            "TestBaseModel",
            __base__=general.CalliopeBaseModel,
            foo=(str, "value1"),
            bar=(int, ...),
        )

    @pytest.fixture(scope="class")
    def dict_base_model(self, base_model):
        dict_model = pydantic.create_model(
            "TestModel",
            __base__=general.CalliopeDictModel,
            root=(dict[str, base_model], {}),
        )
        return dict_model({"key1": {"bar": 2}, "key2": {"foo": "value2", "bar": 3}})

    @pytest.fixture(scope="class")
    def dict_list_model(self, base_model):
        list_model = pydantic.create_model(
            "TestListModel",
            __base__=general.CalliopeListModel,
            root=(list[base_model], []),
        )
        dict_model = pydantic.create_model(
            "TestModel",
            __base__=general.CalliopeDictModel,
            root=(dict[str, list_model], {}),
        )
        return dict_model({"key1": [{"bar": 2}], "key2": [{"foo": "value2", "bar": 3}]})

    def test_no_setitem(self, dict_str_model):
        """Ensure that we cannot set items directly on the CalliopeDictModel."""

        with pytest.raises(general.PydanticCustomError, match="Cannot set a TestModel"):
            dict_str_model["new_key"] = "new_value"

    def test_getitem(self, dict_str_model):
        """Ensure that we can get items from the CalliopeDictModel."""
        assert dict_str_model["key1"] == dict_str_model.root["key1"] == "value1"

    def test_repr(self, dict_str_model):
        """Ensure that the __repr__ method returns the root attribute's __repr__."""
        assert repr(dict_str_model) == repr(dict_str_model.root)

    def test_rich_repr(self, dict_str_model):
        """Ensure that the __rich_repr__ method yields the root attribute's items."""
        rich_repr_items = list(dict_str_model.__rich_repr__())
        assert rich_repr_items == [("key1", "value1"), ("key2", "value2")]

    def test_update(self, dict_str_model):
        """Ensure that the update method returns a new model with updated fields."""
        new_model = dict_str_model.update({"key1": "new_value1", "key3": "value3"})

        assert new_model.root == {
            "key1": "new_value1",
            "key2": "value2",
            "key3": "value3",
        }
        assert dict_str_model.root == {"key1": "value1", "key2": "value2"}

    def test_update_invalid(self, dict_str_model):
        """Ensure that the update method raises an error when it won't pass pydantic validation."""
        with pytest.raises(pydantic.ValidationError, match="1 validation error"):
            dict_str_model.update({"key1": 123})

    def test_check_base_model(self, dict_base_model):
        """Ensure that the CalliopeDictModel can pass updates onto a CalliopeBaseModel."""
        for key in ["key1", "key2"]:
            assert isinstance(dict_base_model.root[key], general.CalliopeBaseModel)
        assert dict_base_model.model_dump() == {
            "key1": {"foo": "value1", "bar": 2},
            "key2": {"foo": "value2", "bar": 3},
        }

    def test_update_base_model(self, dict_base_model):
        """Ensure that the update method works with CalliopeBaseModel values."""
        new_model = dict_base_model.update(
            {"key1": {"foo": "new_value1"}, "key2": {"foo": "new_value2", "bar": 6}}
        )
        for key in ["key1", "key2"]:
            assert isinstance(new_model.root[key], general.CalliopeBaseModel)

        assert new_model.model_dump() == {
            "key1": {"foo": "new_value1", "bar": 2},
            "key2": {"foo": "new_value2", "bar": 6},
        }
        # No change to the original model
        assert dict_base_model.model_dump() == {
            "key1": {"foo": "value1", "bar": 2},
            "key2": {"foo": "value2", "bar": 3},
        }

    def test_update_list_model(self, dict_list_model):
        """Ensure that the update method replaces entire list model."""
        new_model = dict_list_model.update(
            {"key1": [{"bar": 4}], "key2": [{"foo": "new_value2", "bar": 6}]}
        )
        for key in ["key1", "key2"]:
            assert isinstance(new_model[key], general.CalliopeListModel)

        assert new_model.model_dump() == {
            "key1": [{"foo": "value1", "bar": 4}],
            "key2": [{"foo": "new_value2", "bar": 6}],
        }
        # No change to the original model
        assert dict_list_model.model_dump() == {
            "key1": [{"foo": "value1", "bar": 2}],
            "key2": [{"foo": "value2", "bar": 3}],
        }

    def test_log_message_on_adding_dict_entry(self, caplog, dict_str_model):
        """Ensure that a log message is generated when adding a new entry to the CalliopeDictModel."""
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            updated_model = dict_str_model.update({"key3": "value3"})
        assert "Adding TestModel entry: `key3`" in caplog.text

        # Ensure that the new entry was added
        assert updated_model.root["key3"] == "value3"


class TestCalliopeListModel:
    @pytest.fixture(scope="class")
    def list_str_model(self):
        list_model = pydantic.create_model(
            "TestModel", __base__=general.CalliopeListModel, root=(list[str], [])
        )
        return list_model(["value1", "value2"])

    @pytest.fixture(scope="class")
    def list_base_model(self):
        base_model = pydantic.create_model(
            "TestBaseModel",
            __base__=general.CalliopeBaseModel,
            foo=(str, "value1"),
            bar=(int, ...),
        )
        list_model = pydantic.create_model(
            "TestListModel",
            __base__=general.CalliopeListModel,
            root=(list[base_model], []),
        )
        return list_model([{"bar": 2}, {"foo": "value2", "bar": 3}])

    def test_getitem(self, list_str_model):
        """Ensure that we can get items from the CalliopeListModel."""
        assert list_str_model[0] == list_str_model.root[0] == "value1"
        assert list_str_model[1] == list_str_model.root[1] == "value2"

    def test_iter(self, list_str_model):
        """Ensure that we can iterate over the CalliopeListModel."""
        items = list(list_str_model)
        assert items == ["value1", "value2"]

    def test_repr(self, list_str_model):
        """Ensure that the __repr__ method returns the root attribute's __repr__."""
        assert repr(list_str_model) == repr(list_str_model.root)

    def test_rich_repr(self, list_str_model):
        """Ensure that the __rich_repr__ method yields the root attribute's items."""
        rich_repr_items = list(list_str_model.__rich_repr__())
        assert rich_repr_items == ["value1", "value2"]

    def test_update(self, list_str_model):
        """Ensure that the update method returns a new model with updated fields."""
        new_model = list_str_model.update(["new_value1", "new_value2", "new_value3"])

        assert new_model.root == ["new_value1", "new_value2", "new_value3"]
        assert list_str_model.root == ["value1", "value2"]

    def test_update_base_model(self, list_base_model):
        """Ensure that the update method works with CalliopeBaseModel values."""
        new_model = list_base_model.update(
            [{"foo": "new_value1", "bar": 1}, {"foo": "new_value2", "bar": 6}]
        )
        for item in new_model:
            assert isinstance(item, general.CalliopeBaseModel)

        assert new_model.model_dump() == [
            {"foo": "new_value1", "bar": 1},
            {"foo": "new_value2", "bar": 6},
        ]
        # No change to the original model
        assert list_base_model.model_dump() == [
            {"foo": "value1", "bar": 2},
            {"foo": "value2", "bar": 3},
        ]


class TestCalliopeBaseModel:
    @pytest.fixture(scope="module")
    def config_model_flat(self):
        return pydantic.create_model(
            "TestModel1",
            __base__=general.CalliopeBaseModel,
            __config__={"title": "TITLE"},
            foo=(str, "bar"),
            foobar=(int, 1),
        )

    @pytest.fixture(scope="module")
    def config_model_nested(self, config_model_flat):
        return pydantic.create_model(
            "TestModel2",
            __base__=general.CalliopeBaseModel,
            __config__={"title": "TITLE 2"},
            nested=(config_model_flat, config_model_flat()),
            top_level_foobar=(int, 10),
        )

    @pytest.fixture(scope="module")
    def config_model_double_nested(self, config_model_nested):
        return pydantic.create_model(
            "TestModel3",
            __base__=general.CalliopeBaseModel,
            __config__={"title": "TITLE 3"},
            extra_nested=(config_model_nested, config_model_nested()),
        )

    @pytest.fixture(scope="class")
    def config_model_nested_with_dict_and_list_models(self, config_model_flat):
        """Fixture for a nested model with CalliopeDictModel and CalliopeListModel."""
        list_model = pydantic.create_model(
            "TestListModel",
            __base__=general.CalliopeListModel,
            root=(list[config_model_flat], []),
        )
        config_model = pydantic.create_model(
            "TestModel1",
            __base__=general.CalliopeBaseModel,
            __config__={"title": "TITLE"},
            nested_list_field=(list_model, list_model()),
            nested_config=(config_model_flat, config_model_flat()),
            other_field=(str, "default_value"),
        )
        dict_model = pydantic.create_model(
            "TestDictModel",
            __base__=general.CalliopeDictModel,
            root=(dict[str, config_model], {}),
        )
        return pydantic.create_model(
            "TestNestedModel",
            __base__=general.CalliopeBaseModel,
            __config__={"title": "Nested Model"},
            dict_field=(dict_model, dict_model()),
            list_field=(list_model, list_model()),
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

        with pytest.raises(pydantic.ValidationError, match="1 validation error"):
            model.update(to_update)

    def test_update_nested_with_dict_and_list_models(
        self, config_model_nested_with_dict_and_list_models
    ):
        """Test updating a nested model with CalliopeDictModel and CalliopeListModel."""
        model = config_model_nested_with_dict_and_list_models(
            dict_field={
                "key1": {
                    "nested_list_field": [
                        {"foo": "value1", "foobar": 2},
                        {"foo": "value2", "foobar": 4},
                    ],
                    "nested_config": {"foo": "new_value1", "foobar": 1},
                },
                "key2": {"other_field": "value3"},
            },
            list_field=[{"foo": "value3", "foobar": 6}],
        )
        orig_model = model.model_dump()
        # Update the dict field
        new_model = model.update(
            {
                "dict_field": {
                    "key1": {
                        "nested_list_field": [{"foo": "new_value1"}],
                        "nested_config": {"foo": "new_value1"},
                    },
                    "key3": {},
                },
                "list_field": [{}, {"foo": "new_value3", "foobar": 8}],
            }
        )
        assert new_model.model_dump() == {
            "dict_field": {
                "key1": {
                    "nested_list_field": [{"foo": "new_value1", "foobar": 1}],
                    "nested_config": {"foo": "new_value1", "foobar": 1},
                    "other_field": "default_value",
                },
                "key2": {
                    "nested_list_field": [],
                    "nested_config": {"foo": "bar", "foobar": 1},
                    "other_field": "value3",
                },
                "key3": {
                    "nested_list_field": [],
                    "nested_config": {"foo": "bar", "foobar": 1},
                    "other_field": "default_value",
                },
            },
            "list_field": [
                {"foo": "bar", "foobar": 1},
                {"foo": "new_value3", "foobar": 8},
            ],
        }

        # No change in the original model
        assert model.model_dump() == orig_model

    @pytest.mark.parametrize(
        ("to_update", "expected"),
        [
            (
                {"extra_nested.nested.foobar": 2},
                ["Updating TestModel1 `foobar`: 1 -> 2"],
            ),
            (
                {"extra_nested.top_level_foobar": 2},
                ["Updating TestModel2 `top_level_foobar`: 10 -> 2"],
            ),
            (
                {"extra_nested.nested.foobar": 2, "extra_nested.top_level_foobar": 3},
                [
                    "Updating TestModel1 `foobar`: 1 -> 2",
                    "Updating TestModel2 `top_level_foobar`: 10 -> 3",
                ],
            ),
        ],
    )
    def test_logging(self, caplog, config_model_double_nested, to_update, expected):
        model = config_model_double_nested()
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            model.update(to_update)

        assert all(log_text in caplog.text for log_text in expected)

    def test_logging_no_update_on_same_val(
        self, caplog, config_model_nested_with_dict_and_list_models
    ):
        """Ensure that no log message is generated when updating a value to the same value."""

        model = config_model_nested_with_dict_and_list_models(
            dict_field={
                "key1": {"nested_config": {"foo": "value1", "foobar": 1}},
                "key2": {"other_field": "value3"},
            },
            list_field=[{"foo": "value3", "foobar": 6}],
        )
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            model.update(
                {
                    "dict_field.key1.nested_config.foo": "value1",
                    "dict_field.key2.other_field": "value3",
                }
            )
        assert not caplog.text

    def test_config_model_no_defs(self, config_model_nested):
        model = config_model_nested()
        json_schema = model.model_json_schema()
        no_defs_json_schema = model.model_no_ref_schema()
        assert "$defs" in json_schema
        assert "$defs" not in no_defs_json_schema

    def test_config_model_no_resolved_refs(self, config_model_nested):
        model = config_model_nested()
        json_schema = model.model_json_schema()
        no_defs_json_schema = model.model_no_ref_schema()
        assert json_schema["properties"]["nested"] == {
            "$ref": "#/$defs/TestModel1",
            "default": {"foo": "bar", "foobar": 1},
        }
        assert (
            no_defs_json_schema["properties"]["nested"]
            == json_schema["$defs"]["TestModel1"]
        )

    def test_config_model_flat_no_defs(self, config_model_flat):
        """Calling `model_no_ref_schema` should not alter flat models."""
        json_schema = config_model_flat.model_json_schema()
        no_defs_schema = config_model_flat.model_no_ref_schema()

        assert json_schema == no_defs_schema
