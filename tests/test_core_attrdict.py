import pytest

from calliope.attrdict import _MISSING, AttrDict

from .common.util import check_error_or_warning


class TestAttrDict:
    @pytest.fixture
    def regular_dict(self):
        d = {
            "a": 1,
            "b": 2,
            "d": None,
            "c": {"x": "foo", "y": "bar", "z": {"I": 1, "II": 2}},
        }
        return d

    @pytest.fixture
    def attr_dict(self, regular_dict):
        d = regular_dict
        return AttrDict(d)

    def test_missing_nonzero(self):
        assert _MISSING is not True
        assert _MISSING is not False
        assert _MISSING is not None
        assert _MISSING.__nonzero__() is False

    def test_init_from_nondict(self):
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            AttrDict("foo")
        assert check_error_or_warning(excinfo, "Must pass a dict to AttrDict")

    def test_init_from_dict(self, regular_dict):
        d = AttrDict(regular_dict)
        assert d.a == 1

    def test_init_from_dict_with_nested_keys(self):
        d = AttrDict({"foo.bar.baz": 1})
        assert d.foo.bar.baz == 1

    def test_dot_access_first(self, attr_dict):
        d = attr_dict
        assert d.a == 1

    def test_dot_access_second(self, attr_dict):
        d = attr_dict
        assert d.c.x == "foo"

    def test_dot_access_list(self):
        d = AttrDict({"a": [{"x": 1}, {"y": 2}]})
        assert d.a[0].x == 1

    def test_set_key_first(self, attr_dict):
        d = attr_dict
        d.set_key("a", 2)
        assert d.a == 2

    def test_set_key_second(self, attr_dict):
        d = attr_dict
        d.set_key("c.x", "baz")
        assert d.c.x == "baz"

    def test_set_key_multiple_inexisting(self, attr_dict):
        d = attr_dict
        d.set_key("c.l.o.h.a", "foo")
        assert d.c.l.o.h.a == "foo"

    def test_set_key_nested_on_string(self, attr_dict):
        d = attr_dict
        with pytest.raises(KeyError):
            d.set_key("a.foo", "bar")

    def test_set_key_nested_on_none(self, attr_dict):
        d = attr_dict
        assert d["d"] is None
        d.set_key("d.foo", "bar")
        assert d.d.foo == "bar"

    def test_pass_regular_dict_to_set_key(self, attr_dict):
        # Regular dicts get turned into AttrDicts when using
        # assignment through set_key()
        attr_dict.set_key("c.z.newkey", {"foo": 1, "doo": 2})
        assert isinstance(attr_dict.get_key("c.z.newkey"), AttrDict)
        assert attr_dict.get_key("c.z.newkey.foo") == 1

    def test_get_subkey_from_nested_non_attrdict(self, attr_dict):
        # Directly assigning a dict means it is not modified
        # but it breaks get_key with nested keys
        attr_dict["c"]["z"]["newkey"] = {"foo": 1, "doo": 2}

        with pytest.raises(AttributeError) as excinfo:
            attr_dict.get_key("c.z.newkey.foo")

        assert check_error_or_warning(
            excinfo, "'dict' object has no attribute 'get_key'"
        )

    def test_get_key_first(self, attr_dict):
        d = attr_dict
        assert d.get_key("a") == 1

    def test_get_key_second(self, attr_dict):
        d = attr_dict
        assert d.get_key("c.x") == "foo"

    def test_get_key_inexistant(self, attr_dict):
        d = attr_dict
        with pytest.raises(KeyError):
            d.get_key("foo")

    def test_get_key_second_inexistant(self, attr_dict):
        d = attr_dict
        with pytest.raises(KeyError):
            d.get_key("foo.bar")

    def test_get_key_default(self, attr_dict):
        d = attr_dict
        assert d.get_key("c.x", default="bar") == "foo"

    def test_get_key_inexistant_default(self, attr_dict):
        d = attr_dict
        assert d.get_key("foo", default="baz") == "baz"

    def test_get_key_second_inexistant_default(self, attr_dict):
        d = attr_dict
        assert d.get_key("foo.bar", default="baz") == "baz"

    def test_get_key_second_nondict_default(self, attr_dict):
        d = attr_dict
        assert d.get_key("c.x.foo", default="baz") == "baz"

    def test_get_key_inexistant_default_false(self, attr_dict):
        d = attr_dict
        assert d.get_key("foo", default=False) is False

    def test_get_key_second_inexistant_default_false(self, attr_dict):
        d = attr_dict
        assert d.get_key("foo.bar", default=False) is False

    def test_as_dict(self, attr_dict):
        d = attr_dict
        dd = d.as_dict()
        assert dd["a"] == 1
        assert dd["c"]["x"] == "foo"

    def test_as_dict_flat(self, attr_dict):
        dd = attr_dict.as_dict(flat=True)
        assert dd["c.x"] == "foo"

    def test_keys_nested_as_list(self, attr_dict):
        d = attr_dict
        dd = d.keys_nested()
        assert dd == ["a", "b", "d", "c.x", "c.y", "c.z.I", "c.z.II"]

    def test_keys_nested_as_dict(self, attr_dict):
        d = attr_dict
        dd = d.keys_nested(subkeys_as="dict")
        assert dd == ["a", "b", "d", {"c": ["x", "y", {"z": ["I", "II"]}]}]

    def test_union(self, attr_dict):
        d = attr_dict
        d_new = AttrDict()
        d_new.set_key("c.z.III", "foo")
        d.union(d_new)
        assert d.c.z.III == "foo"
        assert d.c.z.I == 1

    def test_union_duplicate_keys(self, attr_dict):
        d = attr_dict
        d_new = AttrDict()
        d_new.set_key("c.z.II", "foo")
        with pytest.raises(KeyError):
            d.union(d_new)

    @pytest.mark.parametrize("to_replace", ["foo", [], {}, 1])
    def test_union_replacement(self, attr_dict, to_replace):
        d = attr_dict
        d_new = AttrDict({"c._REPLACE_": to_replace})
        d.union(d_new, allow_override=True, allow_replacement=True)
        assert d.c == to_replace

    def test_union_empty_dicts(self, attr_dict):
        d = attr_dict
        d_new = AttrDict({"1": {"foo": {}}, "baz": {"bar": {}}})
        d.union(d_new)
        assert len(d.baz.bar.keys()) == 0

    def test_union_order_retained(self, attr_dict):
        d_new = AttrDict({"a": 10, "e": {"b": 1, "a": 2}, "A": -1, "c.z.II": 20})
        attr_dict.union(d_new, allow_override=True)
        assert attr_dict == {
            "a": 10,
            "b": 2,
            "d": None,
            "c": {"x": "foo", "y": "bar", "z": {"I": 1, "II": 20}},
            "e": {"b": 1, "a": 2},
            "A": -1,
        }

    def test_del_key_single(self, attr_dict):
        attr_dict.del_key("c")
        assert "c" not in attr_dict

    def test_del_key_nested(self, attr_dict):
        attr_dict.del_key("c.z.I")
        assert "I" not in attr_dict.c.z
