from io import StringIO
import os

import pytest
import numpy as np
import tempfile
import ruamel.yaml as ruamel_yaml

from calliope.core.attrdict import AttrDict, _MISSING
from calliope.test.common.util import check_error_or_warning


class TestAttrDict:
    @pytest.fixture
    def regular_dict(self):
        d = {'a': 1,
             'b': 2,
             'c': {'x': 'foo',
                   'y': 'bar',
                   'z': {'I': 1,
                         'II': 2}
                   },
             'd': None
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
    def yaml_file(self):
        return StringIO(self.setup_string)

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
        with pytest.raises(ValueError) as excinfo:
            d = AttrDict('foo')
        assert check_error_or_warning(
            excinfo, 'Must pass a dict to AttrDict'
        )

    def test_init_from_dict(self, regular_dict):
        d = AttrDict(regular_dict)
        assert d.a == 1

    def test_init_from_dict_with_nested_keys(self):
        d = AttrDict({'foo.bar.baz': 1})
        assert d.foo.bar.baz == 1

    def test_from_yaml_fobj(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_path(self):
        this_path = os.path.dirname(__file__)
        yaml_path = os.path.join(this_path, 'common', 'yaml_file.yaml')
        d = AttrDict.from_yaml(yaml_path)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string(self, yaml_string):
        d = AttrDict.from_yaml_string(yaml_string)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string_dot_strings(self):
        yaml_string = 'a.b.c: 1\na.b.foo: 2'
        d = AttrDict.from_yaml_string(yaml_string)
        assert d.a.b.c == 1
        assert d.a.b.foo == 2

    def test_from_yaml_string_dot_strings_duplicate(self):
        yaml_string = 'a.b.c: 1\na.b.c: 2'
        d = AttrDict.from_yaml_string(yaml_string)
        assert d.a.b.c == 2

    def test_simple_invalid_yaml(self):
        yaml_string = '1 this is not valid yaml'
        with pytest.raises(ValueError) as excinfo:
            d = AttrDict.from_yaml_string(yaml_string)
        assert check_error_or_warning(
            excinfo, 'Could not parse <yaml string> as YAML'
        )

    def test_parser_error(self):
        with pytest.raises(ruamel_yaml.YAMLError):
            d = AttrDict.from_yaml_string("""
            foo: bar
            baz: 1
                - foobar
                bar: baz

            """)

    def test_dot_access_first(self, attr_dict):
        d = attr_dict
        assert d.a == 1

    def test_dot_access_second(self, attr_dict):
        d = attr_dict
        assert d.c.x == 'foo'

    def test_dot_access_list(self):
        d = AttrDict.from_yaml_string("a: [{x: 1}, {y: 2}]")
        assert d.a[0].x == 1

    def test_set_key_first(self, attr_dict):
        d = attr_dict
        d.set_key('a', 2)
        assert d.a == 2

    def test_set_key_second(self, attr_dict):
        d = attr_dict
        d.set_key('c.x', 'baz')
        assert d.c.x == 'baz'

    def test_set_key_multiple_inexisting(self, attr_dict):
        d = attr_dict
        d.set_key('c.l.o.h.a', 'foo')
        assert d.c.l.o.h.a == 'foo'

    def test_set_key_nested_on_string(self, attr_dict):
        d = attr_dict
        with pytest.raises(KeyError):
            d.set_key('a.foo', 'bar')

    def test_set_key_nested_on_none(self, attr_dict):
        d = attr_dict
        assert d['d'] is None
        d.set_key('d.foo', 'bar')
        assert d.d.foo == 'bar'

    def test_pass_regular_dict_to_set_key(self, attr_dict):
        # Regular dicts get turned into AttrDicts when using
        # assignment through set_key()
        attr_dict.set_key('c.z.newkey', {'foo': 1, 'doo': 2})
        assert isinstance(attr_dict.get_key('c.z.newkey'), AttrDict)
        assert attr_dict.get_key('c.z.newkey.foo') == 1

    def test_get_subkey_from_nested_non_attrdict(self, attr_dict):
        # Directly assigning a dict means it is not modified
        # but it breaks get_key with nested keys
        attr_dict['c']['z']['newkey'] = {'foo': 1, 'doo': 2}

        with pytest.raises(AttributeError) as excinfo:
            attr_dict.get_key('c.z.newkey.foo')

        assert check_error_or_warning(
            excinfo, "'dict' object has no attribute 'get_key'"
        )

    def test_get_key_first(self, attr_dict):
        d = attr_dict
        assert d.get_key('a') == 1

    def test_get_key_second(self, attr_dict):
        d = attr_dict
        assert d.get_key('c.x') == 'foo'

    def test_get_key_inexistant(self, attr_dict):
        d = attr_dict
        with pytest.raises(KeyError):
            d.get_key('foo')

    def test_get_key_second_inexistant(self, attr_dict):
        d = attr_dict
        with pytest.raises(KeyError):
            d.get_key('foo.bar')

    def test_get_key_default(self, attr_dict):
        d = attr_dict
        assert d.get_key('c.x', default='bar') == 'foo'

    def test_get_key_inexistant_default(self, attr_dict):
        d = attr_dict
        assert d.get_key('foo', default='baz') == 'baz'

    def test_get_key_second_inexistant_default(self, attr_dict):
        d = attr_dict
        assert d.get_key('foo.bar', default='baz') == 'baz'

    def test_get_key_second_nondict_default(self, attr_dict):
        d = attr_dict
        assert d.get_key('c.x.foo', default='baz') == 'baz'

    def test_get_key_inexistant_default_false(self, attr_dict):
        d = attr_dict
        assert d.get_key('foo', default=False) is False

    def test_get_key_second_inexistant_default_false(self, attr_dict):
        d = attr_dict
        assert d.get_key('foo.bar', default=False) is False

    def test_as_dict(self, attr_dict):
        d = attr_dict
        dd = d.as_dict()
        assert dd['a'] == 1
        assert dd['c']['x'] == 'foo'

    def test_as_dict_with_sublists(self):
        d = AttrDict.from_yaml_string("a: [{x: 1}, {y: 2}]")
        dd = d.as_dict()
        assert dd['a'][0]['x'] == 1
        assert isinstance(dd['a'][0], dict)  # Not AttrDict!

    def test_as_dict_flat(self, attr_dict):
        dd = attr_dict.as_dict(flat=True)
        assert dd['c.x'] == 'foo'

    def test_keys_nested_as_list(self, attr_dict):
        d = attr_dict
        dd = d.keys_nested()
        assert dd == ['a', 'b', 'c.x', 'c.y', 'c.z.I', 'c.z.II', 'd']

    def test_keys_nested_as_dict(self, attr_dict):
        d = attr_dict
        dd = d.keys_nested(subkeys_as='dict')
        assert dd == ['a', 'b', {'c': ['x', 'y', {'z': ['I', 'II']}]}, 'd']

    def test_union(self, attr_dict):
        d = attr_dict
        d_new = AttrDict()
        d_new.set_key('c.z.III', 'foo')
        d.union(d_new)
        assert d.c.z.III == 'foo'
        assert d.c.z.I == 1

    def test_union_duplicate_keys(self, attr_dict):
        d = attr_dict
        d_new = AttrDict()
        d_new.set_key('c.z.II', 'foo')
        with pytest.raises(KeyError):
            d.union(d_new)

    def test_union_replacement(self, attr_dict):
        d = attr_dict
        d_new = AttrDict.from_yaml_string("""
            c: {_REPLACE_: foo}
        """)
        d.union(d_new, allow_override=True, allow_replacement=True)
        assert d.c == 'foo'

    def test_union_empty_dicts(self, attr_dict):
        d = attr_dict
        d_new = AttrDict({
            '1': {'foo': {}},
            'baz': {'bar': {}},
        })
        d.union(d_new)
        assert d.baz.bar.keys() == []

    def test_del_key_single(self, attr_dict):
        attr_dict.del_key('c')
        assert 'c' not in attr_dict

    def test_del_key_nested(self, attr_dict):
        attr_dict.del_key('c.z.I')
        assert 'I' not in attr_dict.c.z

    def test_to_yaml(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        d.set_key('numpy.some_int', np.int32(10))
        d.set_key('numpy.some_float', np.float64(0.5))
        d.a_list = [0, 1, 2]
        with tempfile.TemporaryDirectory() as tempdir:
            out_file = os.path.join(tempdir, 'test.yaml')
            d.to_yaml(out_file)

            with open(out_file, 'r') as f:
                result = f.read()

            assert 'some_int: 10' in result
            assert 'some_float: 0.5' in result
            assert 'a_list:\n- 0\n- 1\n- 2' in result

    def test_to_yaml_string(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        result = d.to_yaml()
        assert 'a: 1' in result

    def test_comment_roundtrip(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        with tempfile.TemporaryDirectory() as tempdir:
            out_file = os.path.join(tempdir, 'test.yaml')
            d.to_yaml(out_file)

            with open(out_file, 'r') as f:
                result = f.read()

            assert 'x: foo  # a comment on foo' in result

    def test_iter_with_comments_filtered(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        keys = [k for k in d]
        assert '__dict_comments__' not in keys

    def test_keys_with_comments_not_filtered(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        keys = d.keys(filtered=False)
        assert '__dict_comments__' in keys

    def test_values_with_comments_not_filtered(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        values = d.values(filtered=False)
        c_value = [i for i in values if isinstance(i, dict) and 'c' in i]
        assert c_value[0]['c'][0] is None

    def test_keys_nested_with_comments_flat(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        dd = d.keys_nested()
        assert dd == ['a', 'b', 'c.x', 'c.y', 'c.z.I', 'c.z.II', 'd']

    def test_keys_nested_with_comments_as_dict(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        dd = d.keys_nested(subkeys_as='dict')
        assert dd == ['a', 'b', {'c': ['x', 'y', {'z': ['I', 'II']}]}, 'd']

    def test_add_comment_invalid_kind(self, attr_dict):
        with pytest.raises(ValueError):
            attr_dict.set_comment('foo', 'bar', kind='baz')

    def test_add_comment(self, attr_dict):
        attr_dict.set_comment('c.y', 'test comment')  # inline
        attr_dict.set_comment('c.y', 'test comment above', kind='above')

        wanted_comments = [
            None,
            [ruamel_yaml.CommentToken('# test comment above', 0, None)],
            ruamel_yaml.CommentToken('# test comment', 0, None),
            None
        ]

        resulting_comments = attr_dict.c.__dict_comments__['y']
        assert resulting_comments[1][0].value == wanted_comments[1][0].value
        assert resulting_comments[2].value == wanted_comments[2].value

    def test_add_comment_exists(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)

        # should work, as no inline comment exists
        d.set_comment('b', 'test comment')

        # should fail, as an above comment exists
        with pytest.raises(ValueError) as excinfo:
            d.set_comment('c', 'test comment', kind='above')

        assert check_error_or_warning(
            excinfo, 'Comment exists'
        )

    def test_add_comment_exists_but_empty(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)

        # Ensure comments were added
        assert d.c.__dict_comments__['y'][2].value == '#\n'

        # should work, as inline comment is empty
        d.set_comment('c.y', 'test comment')

        # should work, as above comment is empty
        d.set_comment('c.y', 'test comment', kind='above')

    def test_del_comment(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        assert 'y' in d.c.__dict_comments__
        d.del_comments('c.y')
        assert 'y' not in d.c.__dict_comments__

    def test_get_comment(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        result = {
            'above': '# a comment about `c`\n',
            'inline': '# a comment inline with `c`\n',
            'below': None
        }
        assert d.get_comments('c') == result

    def test_get_comment_inexistent(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        with pytest.raises(KeyError):
            d.get_comments('b')

    def test_get_comment_all_none(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        d.__dict_comments__['b'] = [None, None, None, None]
        result = {
            'above': None,
            'inline': None,
            'below': None
        }
        assert d.get_comments('b') == result

    def test_delete_key_deletes_comments(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        assert 'x' in d.c.__dict_comments__
        d.del_key('c.x')
        assert 'x' not in d.c.__dict_comments__

    def test_union_preserves_comments(self, yaml_file):
        d = AttrDict.from_yaml(yaml_file)
        d_new = AttrDict.from_yaml_string("""
            test: 1  # And a comment
            somekey:
                bar:
                    baz: 2  # Another comment
        """)
        d.union(d_new)
        assert d.get_comments('somekey.bar.baz')['inline'] == '# Another comment\n'

    def test_import_preserves_comments(self, yaml_file):
        with tempfile.TemporaryDirectory() as tempdir:
            imported_file = os.path.join(tempdir, 'test_import.yaml')
            imported_yaml = """
                somekey: 1
                anotherkey: 2  # anotherkey's comment
            """
            with open(imported_file, 'w') as f:
                f.write(imported_yaml)

            yaml_string = """
                import:
                    - {}
                foo:
                    bar: 1  # Comment on bar
                    baz: 2
                    3:
                        4: 5
            """.format(imported_file)

            d = AttrDict.from_yaml_string(yaml_string, resolve_imports=True)

        assert 'anotherkey' in d.__dict_comments__

        assert d.get_comments('anotherkey')['inline'] == "# anotherkey's comment\n"

    def test_comment_on_collapsed_dict_notation(self):
        yaml_string = """
            foo:
                bar: 1  # Comment on bar
                baz: 2
                3:
                    4: 5
            baz.bar.foo.doo: 4  # Big bad comment
        """

        d = AttrDict.from_yaml_string(yaml_string)
        d_flat = d.as_dict_flat(filtered=False)
        assert d_flat['__dict_comments__'] == {}
        assert 'doo' in d_flat['baz.bar.foo.__dict_comments__']
