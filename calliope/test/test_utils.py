from __future__ import print_function
from __future__ import division

import cStringIO as StringIO
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
        return StringIO.StringIO(self.setup_string)

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

    def test_dot_access_firstl(self, attr_dict):
        d = attr_dict
        assert d.a == 1

    def test_dot_access_second(self, attr_dict):
        d = attr_dict
        assert d.c.x == 'foo'

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

    def test_keys_nested_as_list(self, attr_dict):
        d = attr_dict
        dd = d.keys_nested()
        assert dd == ['a', 'b', 'c.x', 'c.y', 'c.z.I', 'c.z.II', 'd']

    def test_keys_nested_as_dict(self, attr_dict):
        d = attr_dict
        dd = d.keys_nested(subkeys_as='dict')
        # The sort order is: dicts first, then string keys
        assert dd == [{'c': [{'z': ['I', 'II']}, 'x', 'y']}, 'a', 'b', 'd']

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


class TestReplace:
    @pytest.fixture
    def path_list(self):
        return ['{{ foo }}', '{{foo}}', '{{foo }}', '{{ foo}}',
                '{{ bar }}', '{{ foo }}/{{ foo }}/bar']

    def test_replace(self, path_list):
        assert utils.replace(path_list[0], 'foo', 'baz') == 'baz'
        assert utils.replace(path_list[1], 'foo', 'baz') == 'baz'

    def test_replace_unbalanced(self, path_list):
        assert utils.replace(path_list[2], 'foo', 'baz') == '{{foo }}'
        assert utils.replace(path_list[3], 'foo', 'baz') == '{{ foo}}'

    def test_replace_not_string(self, path_list):
        with pytest.raises(AttributeError):
            utils.replace(10, 'foo', 'baz')

    def test_replace_not_placeholder(self, path_list):
        assert utils.replace(path_list[4], 'foo', 'baz') == '{{ bar }}'

    def test_replace_multiple(self, path_list):
        assert utils.replace(path_list[5], 'foo', 'baz') == 'baz/baz/bar'
