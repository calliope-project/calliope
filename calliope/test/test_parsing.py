from io import StringIO

import pytest
import ruamel.yaml as yaml
import pyparsing as pp
import xarray as xr

from calliope.backend import parsing
from calliope.test.common.util import check_error_or_warning


def string_to_dict(yaml_string):
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    return yaml_loader.load(StringIO(yaml_string))


def constraint_string(equation_expr):
    setup_string = f"""
    foreach: [a in A, a1 in A1]
    where: []
    equation: "{equation_expr}"
    """

    return string_to_dict(setup_string)


@pytest.fixture
def dummy_model_data():
    d = {
        "A": {"dims": ("A"), "data": [1, 2]},
        "A1": {"dims": ("A1"), "data": [10, 20, 30]},
        "foo": {"dims": ("A", "A1"), "data": [["a", "b", "c"], ["d", "e", "f"]]},
        "bar": {"dims": ("A"), "data": [100, 200]},
    }
    return xr.Dataset.from_dict(d)


@pytest.fixture
def dummy_model_data():
    d = {
        "A": {"dims": ("A"), "data": [1, 2]},
        "A1": {"dims": ("A1"), "data": [10, 20, 30]},
        "A_1": {"dims": ("A_1"), "data": [-1, -2, -3]},
        "techs": {"dims": ("techs"), "data": ["foo1", "bar1", "foobar1"]},
        "foo": {"dims": ("A", "A1"), "data": [["a", "b", "c"], ["d", "e", "f"]]},
        "bar": {"dims": ("A"), "data": [100, 200]},
    }
    return xr.Dataset.from_dict(d)


@pytest.fixture
def dummy_constraint_obj():
    constraint_data = constraint_string("1 == 1")
    return parsing.ParsedConstraint(constraint_data, "foo")


class TestParsingForEach:
    @pytest.fixture(
        params=[
            ("[a in A]", ["a"], ["A"], []),
            ("[a in A, a1 in A1]", ["a", "a1"], ["A", "A1"], []),
            ("[a in A, a_2 in A_2]", ["a"], ["A"], ["A_1"]),
            ("[a in A, a_2 in A_2, foo in foos]", ["a"], ["A"], ["A_2", "foos"]),
        ]
    )
    def constraint_data(self, request, dummy_model_data):
        foreach_string, set_iterators, set_names, missing_sets = request.param
        setup_string = f"""
        foreach: {foreach_string}
        where: []
        equation: foo{set_iterators} == 0
        """
        constraint_obj = parsing.ParsedConstraint(string_to_dict(setup_string), "foo")
        constraint_obj._get_sets_from_foreach(dummy_model_data.dims)
        return (
            constraint_obj,
            set_iterators,
            set_names,
            missing_sets,
        )

    @pytest.fixture
    def foreach_parser(self, dummy_constraint_obj):
        return dummy_constraint_obj._foreach_parser()

    @pytest.mark.parametrize(
        ("input_string", "expected_result"),
        [
            ("a in A", ["a", "A"]),
            ("a1 in A1", ["a1", "A1"]),
            ("a_1 in A_1", ["a_1", "A_1"]),
            ("tech in techs", ["tech", "techs"]),
            # TODO: decide if this should be allowed:
            ("techs in techs", ["techs", "techs"]),
        ],
    )
    def test_parse_foreach(self, foreach_parser, input_string, expected_result):
        parsed_string = foreach_parser.parse_string(input_string, parse_all=True)
        assert parsed_string.as_dict() == {
            "set_iterator": expected_result[0],
            "set_name": expected_result[1],
        }

    @pytest.mark.parametrize(
        "input_string",
        [
            "1 in foo",  # number as iterator
            "foo in 1",  # number as set name
            "1 in 2",  # numbers for both
            "in B",  # missing iterator
            "in",  # missing iterator and set name
            "foo bar",  # missing "in"
            "foo.bar in B",  # unallowed character in iterator .
            "a in foo.bar",  # unallowed character in set name .
            "ainA",  # missing whitespace
            "1a in 2b",  # invalid python identifiers
            "a in A b in B",  # missing deliminator between two set items
            "a in in A",  # duplicated "in"
            "a in in"  # Cannot have "in" as a set iterator/name
            "in in A"  # Cannot have "in" as a set iterator/name
            "in in in",  # Cannot have "in" as a set iterator/name
        ],
    )
    def test_parse_foreach_fail(self, foreach_parser, input_string):
        with pytest.raises(pp.ParseException):
            foreach_parser.parse_string(input_string, parse_all=True)

    def test_parse_foreach_to_sets(self, constraint_data):
        (
            constraint_obj,
            expected_set_iterator,
            expected_set_names,
            _,
        ) = constraint_data
        assert set(constraint_obj.sets.keys()) == set(expected_set_iterator)
        assert set(constraint_obj.sets.values()) == set(expected_set_names)

    def test_parse_foreach_to_sets_unknown_set(self, constraint_data):
        constraint_obj, _, _, missing_sets = constraint_data

        if len(missing_sets) == 0:
            assert constraint_obj._errors == []
        else:
            assert check_error_or_warning(
                constraint_obj._errors, "not a valid model set name."
            )

    def test_parse_foreach_duplicate_iterators(self):
        setup_string = f"""
        foreach: [a in A, a in A1]
        where: []
        equation: foo == 0
        """
        constraint_obj = parsing.ParsedConstraint(string_to_dict(setup_string), "foo")
        constraint_obj._get_sets_from_foreach(["A", "A1"])
        assert check_error_or_warning(
            constraint_obj._errors,
            "(foreach, a in A1): Found duplicate set iterator `a`",
        )
