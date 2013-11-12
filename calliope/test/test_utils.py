from __future__ import print_function
from __future__ import division

import pytest
import sys

import lisa


class TestAttrDict:
    @pytest.fixture
    def regular_dict(self):
        d = {'a': 1,
             'b': 2,
             'c': {'x': 'foo',
                   'y': 'bar',
                   'z': {'I': 1,
                         'II': 2}
                   }
             }
        return d

    @pytest.fixture
    def attr_dict(self, regular_dict):
        d = regular_dict
        return lisa.utils.AttrDict(d)

    def test_init_from_dict(self, regular_dict):
        d = lisa.utils.AttrDict(regular_dict)
        assert d.a == 1

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


class TestCaptureOutput():
    def example_funct(self):
        print('the first thing')
        print('the second thing')
        print('the error thing', file=sys.stderr)

    def test_capture_output(self):
        with lisa.utils.capture_output() as out:
            self.example_funct()
        assert out[0] == 'the first thing\nthe second thing\n'
        assert out[1] == 'the error thing\n'
