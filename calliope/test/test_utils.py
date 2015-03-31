from io import StringIO
import os
import pytest
import sys

from calliope import utils


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
        a: 1
        b: 2
        c:
            x: foo
            y: bar
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
        return utils.AttrDict(d)

    def test_init_from_dict(self, regular_dict):
        d = utils.AttrDict(regular_dict)
        assert d.a == 1

    def test_init_from_dict_with_nested_keys(self):
        d = utils.AttrDict({'foo.bar.baz': 1})
        assert d.foo.bar.baz == 1

    def test_from_yaml_fobj(self, yaml_file):
        d = utils.AttrDict.from_yaml(yaml_file)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_path(self):
        this_path = os.path.dirname(__file__)
        yaml_path = os.path.join(this_path, 'common', 'yaml_file.yaml')
        d = utils.AttrDict.from_yaml(yaml_path)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string(self, yaml_string):
        d = utils.AttrDict.from_yaml_string(yaml_string)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string_dot_strings(self):
        yaml_string = 'a.b.c: 1\na.b.foo: 2'
        d = utils.AttrDict.from_yaml_string(yaml_string)
        assert d.a.b.c == 1
        assert d.a.b.foo == 2

    def test_from_yaml_string_dot_strings_duplicate(self):
        yaml_string = 'a.b.c: 1\na.b.c: 2'
        d = utils.AttrDict.from_yaml_string(yaml_string)
        assert d.a.b.c == 2

    def test_dot_access_first(self, attr_dict):
        d = attr_dict
        assert d.a == 1

    def test_dot_access_second(self, attr_dict):
        d = attr_dict
        assert d.c.x == 'foo'

    def test_dot_access_list(self):
        d = utils.AttrDict.from_yaml_string("a: [{x: 1}, {y: 2}]")
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
        d = utils.AttrDict.from_yaml_string("a: [{x: 1}, {y: 2}]")
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
        d_new = utils.AttrDict()
        d_new.set_key('c.z.III', 'foo')
        d.union(d_new)
        assert d.c.z.III == 'foo'
        assert d.c.z.I == 1

    def test_union_duplicate_keys(self, attr_dict):
        d = attr_dict
        d_new = utils.AttrDict()
        d_new.set_key('c.z.II', 'foo')
        with pytest.raises(KeyError):
            d.union(d_new)

    def test_union_replacement(self, attr_dict):
        d = attr_dict
        d_new = utils.AttrDict.from_yaml_string("""
            c: {_REPLACE_: foo}
        """)
        d.union(d_new, allow_override=True, allow_replacement=True)
        assert d.c == 'foo'


class TestCaptureOutput:
    def example_funct(self):
        print('the first thing')
        print('the second thing')
        print('the error thing', file=sys.stderr)

    def test_capture_output(self):
        with utils.capture_output() as out:
            self.example_funct()
        assert out[0] == 'the first thing\nthe second thing\n'
        assert out[1] == 'the error thing\n'


class TestMemoization:
    @utils.memoize_instancemethod
    def instance_method(self, a, b):
        return a + b

    def test_memoize_one_arg(self):
        @utils.memoize
        def test(a):
            return a + 1
        assert test(1) == 2
        assert test(1) == 2

    def test_memoize_two_args(self):
        def test(a, b):
            return a + b
        assert test(1, 2) == 3
        assert test(1, 2) == 3

    def test_memoize_instancemethod(self):
        assert self.instance_method(1, 2) == 3
        assert self.instance_method(1, 2) == 3
