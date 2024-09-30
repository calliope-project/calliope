import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import ruamel.yaml as ruamel_yaml

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

    setup_string = """
        # a comment
        a: 1
        b: 2
        # a comment about `c`
        c:  # a comment inline with `c`
            x: foo  # a comment on foo

            #
            y: bar  #
            z:
                I: 1
                II: 2
        d:
    """

    @pytest.fixture
    def yaml_filepath(self):
        this_path = Path(__file__).parent
        return this_path / "common" / "yaml_file.yaml"

    @pytest.fixture
    def yaml_string(self):
        return self.setup_string

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

    def test_from_yaml_path(self, yaml_filepath):
        d = AttrDict.from_yaml(yaml_filepath)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string(self, yaml_string):
        d = AttrDict.from_yaml_string(yaml_string)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string_dot_strings(self):
        yaml_string = "a.b.c: 1\na.b.foo: 2"
        d = AttrDict.from_yaml_string(yaml_string)
        assert d.a.b.c == 1
        assert d.a.b.foo == 2

    def test_from_yaml_string_dot_strings_duplicate(self):
        yaml_string = "a.b.c: 1\na.b.c: 2"
        with pytest.raises(ruamel_yaml.constructor.DuplicateKeyError):
            AttrDict.from_yaml_string(yaml_string)

    def test_simple_invalid_yaml(self):
        yaml_string = "1 this is not valid yaml"
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            AttrDict.from_yaml_string(yaml_string)
        assert check_error_or_warning(excinfo, "Could not parse <yaml string> as YAML")

    def test_parser_error(self):
        with pytest.raises(ruamel_yaml.YAMLError):
            AttrDict.from_yaml_string(
                """
            foo: bar
            baz: 1
                - foobar
                bar: baz

            """
            )

    def test_order_of_subdicts(self):
        d = AttrDict.from_yaml_string(
            """
            A.B.C: 10
            A.B:
                E: 20
        """
        )
        assert d.A.B.C == 10
        assert d.A.B.E == 20

    def test_dot_access_first(self, attr_dict):
        d = attr_dict
        assert d.a == 1

    def test_dot_access_second(self, attr_dict):
        d = attr_dict
        assert d.c.x == "foo"

    def test_dot_access_list(self):
        d = AttrDict.from_yaml_string("a: [{x: 1}, {y: 2}]")
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

    def test_as_dict_with_sublists(self):
        d = AttrDict.from_yaml_string("a: [{x: 1}, {y: 2}]")
        dd = d.as_dict()
        assert dd["a"][0]["x"] == 1
        assert isinstance(dd["a"][0], dict)  # Not AttrDict!

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
        d_new = AttrDict.from_yaml_string(f"c._REPLACE_: {to_replace}")
        d.union(d_new, allow_override=True, allow_replacement=True)
        assert d.c == to_replace

    def test_union_replacement_null(self, attr_dict):
        d = attr_dict
        d_new = AttrDict.from_yaml_string("c._REPLACE_: null")
        d.union(d_new, allow_override=True, allow_replacement=True)
        assert d.c is None

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

    def test_to_yaml(self, yaml_filepath):
        d = AttrDict.from_yaml(yaml_filepath)
        d.set_key("numpy.some_int", np.int32(10))
        d.set_key("numpy.some_float", np.float64(0.5))
        d.a_list = [0, 1, 2]
        with tempfile.TemporaryDirectory() as tempdir:
            out_file = os.path.join(tempdir, "test.yaml")
            d.to_yaml(out_file)

            with open(out_file) as f:
                result = f.read()

            assert "some_int: 10" in result
            assert "some_float: 0.5" in result
            assert "a_list:\n- 0\n- 1\n- 2" in result

    def test_to_yaml_string(self, yaml_filepath):
        d = AttrDict.from_yaml(yaml_filepath)
        result = d.to_yaml()
        assert "a: 1" in result

    def test_import_must_be_list(self):
        yaml_string = """
            import: 'somefile.yaml'
        """
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            AttrDict.from_yaml_string(yaml_string, resolve_imports=True)
        assert check_error_or_warning(excinfo, "`import` must be a list.")

    def test_do_not_resolve_imports(self):
        yaml_string = """
            import: ['somefile.yaml']
        """
        d = AttrDict.from_yaml_string(yaml_string, resolve_imports=False)
        # Should not raise an error about a missing file, as we ask for
        # imports not to be resolved
        assert d["import"] == ["somefile.yaml"]

    def test_nested_import(self):
        with tempfile.TemporaryDirectory() as tempdir:
            imported_file = os.path.join(tempdir, "test_import.yaml")
            imported_yaml = """
                somekey: 1
                anotherkey: 2
            """
            with open(imported_file, "w") as f:
                f.write(imported_yaml)

            yaml_string = f"""
                foobar:
                    import:
                        - {imported_file}
                foo:
                    bar: 1
                    baz: 2
                    3:
                        4: 5
            """

            d = AttrDict.from_yaml_string(yaml_string, resolve_imports="foobar")

        assert "foobar.somekey" in d.keys_nested()
        assert d.get_key("foobar.anotherkey") == 2
