from io import StringIO

import pytest
import ruamel.yaml as yaml
import pyparsing
import xarray as xr

from calliope.backend import parsing


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


class TestParsingForEach:
    @pytest.fixture(
        params=[
            ("[a in A]", ["a"], ["A"], []),
            ("[a in A, a1 in A1]", ["a", "a1"], ["A", "A1"], []),
            ("[a in A, a_1 in A_1]", ["a", "a_1"], ["A", "A_1"], ["A_1"]),
            (
                "[a in A, a1 in A1, a_1 in A_1, foo in foos]",
                ["a", "a1", "a_1", "foo"],
                ["A", "A1", "A_1", "foos"],
                ["A_1", "foos"],
            ),  # TODO: test set_iterator == set_name?
        ]
    )
    def constraint_data(self, request):
        foreach_string, set_iterators, set_names, missing_sets = request.param
        setup_string = f"""
        foreach: {foreach_string}
        where: []
        equation: foo[{set_iterators}] == 0
        """
        return (
            string_to_dict(setup_string),
            foreach_string,
            set_iterators,
            set_names,
            missing_sets,
        )

    @pytest.mark.parametrize(
        ("input_string", "expected_result"),
        [
            ("a in A", ["a", "A"]),
            ("a1 in A2", ["a1", "A2"]),
            ("a_1 in A_2", ["a_1", "A_2"]),
            ("foo in foos", ["foo", "foos"]),
        ],  # TODO: test set_iterator == set_name?  # TODO: what should happen on e.g. `in in in`?
    )
    def test_parse_foreach(self, input_string, expected_result):
        parsed_string = parsing.parse_foreach(input_string)
        assert parsed_string == {
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
        ],
    )
    def test_parse_foreach_fail(self, input_string):
        with pytest.raises(pyparsing.ParseException):
            parsing.parse_foreach(input_string)

    def test_parse_foreach_to_sets(self, constraint_data, dummy_model_data):
        (
            constraint_string,
            _,
            expected_set_iterator,
            expected_set_names,
            _,
        ) = constraint_data
        constraint_obj = parsing.ParsedConstraint(
            constraint_string, "foobar", dummy_model_data
        )
        assert constraint_obj.set_iterators == expected_set_iterator
        assert constraint_obj.set_names == expected_set_names

    def test_parse_foreach_to_sets_unknown_set(self, constraint_data, dummy_model_data):
        constraint_string, _, _, _, missing_sets = constraint_data
        constraint_obj = parsing.ParsedConstraint(
            constraint_string, "foo", dummy_model_data
        )
        if len(missing_sets) == 0:
            assert constraint_obj._errors == []
        else:
            assert constraint_obj._errors == [
                f"Constraint sets {set(missing_sets)} must be given as dimensions in the model dataset"
            ]
