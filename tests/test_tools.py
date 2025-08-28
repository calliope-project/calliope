import pytest

from calliope.util import tools


class TestListify:
    @pytest.mark.parametrize(
        ("var", "expected"), [(True, [True]), (1, [1]), ("foobar", ["foobar"])]
    )
    def test_non_iterable(self, var, expected):
        """Listification should work for any kind of object."""
        assert tools.listify(var) == expected

    @pytest.mark.parametrize(
        ("var", "expected"),
        [([1, 2, 3, 4], [1, 2, 3, 4]), ({"foo": "bar", "bar": "foo"}, ["foo", "bar"])],
    )
    def test_iterable(self, var, expected):
        """Iterable objects should be returned as lists."""
        assert tools.listify(var) == expected

    @pytest.mark.parametrize(("var", "expected"), [([], []), (None, []), ({}, [])])
    def test_empty(self, var, expected):
        """Empty iterables, None and similars should be returned as an empty list."""
        assert tools.listify(var) == expected


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("init.datetime_format", "ISO8601"),
        ("build.backend", "pyomo"),
        ("build.operate.window", "24h"),
        ("init.pre_validate_math_strings", False),
    ],
)
class TestDotAttr:
    def test_pydantic_access(self, default_config, attr, expected):
        """Dot access of pydantic attributes should be possible."""
        assert tools.get_dot_attr(default_config, attr) == expected

    def test_dict_access(self, default_config, attr, expected):
        """Dot access of dictionary items should be possible."""
        config_dict = default_config.model_dump()
        assert tools.get_dot_attr(config_dict, attr) == expected
