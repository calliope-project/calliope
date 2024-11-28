import os
import tempfile
from pathlib import Path

import numpy as np
import pytest  # noqa: F401
import ruamel.yaml as ruamel_yaml
import xarray as xr

import calliope
import calliope.io
from calliope import exceptions
from calliope.attrdict import AttrDict

from .common.util import check_error_or_warning


class TestIO:
    @pytest.fixture(scope="module")
    def vars_to_add_attrs(self):
        return ["source_use_max", "flow_cap"]

    @pytest.fixture(scope="module")
    def model(self, vars_to_add_attrs):
        model = calliope.examples.national_scale()
        attrs = {
            "foo_true": True,
            "foo_false": False,
            "foo_none": None,
            "foo_dict": {"foo": {"a": 1}},
            "foo_attrdict": calliope.AttrDict({"foo": {"a": 1}}),
            "foo_set": set(["foo", "bar"]),
            "foo_set_1_item": set(["foo"]),
            "foo_list": ["foo", "bar"],
            "foo_list_1_item": ["foo"],
        }
        model._model_data = model._model_data.assign_attrs(**attrs)
        model.build()
        model.solve()

        for var in vars_to_add_attrs:
            model._model_data[var] = model._model_data[var].assign_attrs(**attrs)

        return model

    @pytest.fixture(scope="module")
    def model_file(self, tmpdir_factory, model):
        out_path = tmpdir_factory.mktemp("data").join("model.nc")
        model.to_netcdf(out_path)
        return out_path

    @pytest.fixture(scope="module")
    def model_from_file(self, model_file):
        return calliope.read_netcdf(model_file)

    @pytest.fixture(scope="module")
    def model_from_file_no_processing(self, model_file):
        return xr.open_dataset(model_file)

    @pytest.fixture(scope="module")
    def model_csv_dir(self, tmpdir_factory, model):
        out_path = tmpdir_factory.mktemp("data")
        model.to_csv(os.path.join(out_path, "csvs"))
        return os.path.join(out_path, "csvs")

    def test_save_netcdf(self, model_file):
        assert os.path.isfile(model_file)

    @pytest.mark.parametrize(
        ("attr", "expected_type", "expected_val"),
        [
            ("foo_true", bool, True),
            ("foo_false", bool, False),
            ("foo_none", type(None), None),
            ("foo_dict", dict, {"foo": {"a": 1}}),
            ("foo_attrdict", calliope.AttrDict, calliope.AttrDict({"foo": {"a": 1}})),
            ("foo_set", set, set(["foo", "bar"])),
            ("foo_set_1_item", set, set(["foo"])),
            ("foo_list", list, ["foo", "bar"]),
            ("foo_list_1_item", list, ["foo"]),
        ],
    )
    @pytest.mark.parametrize("model_name", ["model", "model_from_file"])
    def test_serialised_attrs(
        self, request, attr, expected_type, expected_val, model_name, vars_to_add_attrs
    ):
        model = request.getfixturevalue(model_name)
        var_attrs = [model._model_data[var].attrs for var in vars_to_add_attrs]
        for attrs in [model._model_data.attrs, *var_attrs]:
            assert isinstance(attrs[attr], expected_type)
            if expected_val is None:
                assert attrs[attr] is None
            else:
                assert attrs[attr] == expected_val

    @pytest.mark.parametrize(
        "serialised_list",
        ["serialised_bools", "serialised_nones", "serialised_dicts", "serialised_sets"],
    )
    @pytest.mark.parametrize("model_name", ["model", "model_from_file"])
    def test_serialised_list_popped(self, request, serialised_list, model_name):
        model = request.getfixturevalue(model_name)
        assert serialised_list not in model._model_data.attrs.keys()

    @pytest.mark.parametrize(
        ("serialisation_list_name", "list_elements"),
        [
            ("serialised_bools", ["foo_true", "foo_false"]),
            ("serialised_nones", ["foo_none", "scenario"]),
            (
                "serialised_dicts",
                ["foo_dict", "foo_attrdict", "defaults", "config", "applied_math"],
            ),
            ("serialised_sets", ["foo_set", "foo_set_1_item"]),
            ("serialised_single_element_list", ["foo_list_1_item", "foo_set_1_item"]),
        ],
    )
    def test_serialisation_lists(
        self, model_from_file_no_processing, serialisation_list_name, list_elements
    ):
        serialisation_list = calliope.io._pop_serialised_list(
            model_from_file_no_processing.attrs, serialisation_list_name
        )
        assert not set(serialisation_list).symmetric_difference(list_elements)

    @pytest.mark.parametrize(
        "attrs", [{"foo": [1]}, {"foo": [None]}, {"foo": [1, "bar"]}, {"foo": set([1])}]
    )
    def test_non_strings_in_serialised_lists(self, attrs):
        with pytest.raises(TypeError) as excinfo:
            calliope.io._serialise(attrs)
        assert check_error_or_warning(
            excinfo,
            f"Cannot serialise a sequence of values stored in a model attribute unless all values are strings, found: {attrs['foo']}",
        )

    def test_save_csv_dir_mustnt_exist(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir)
            with pytest.raises(FileExistsError):
                model.to_csv(out_path)

    def test_save_csv_dir_can_exist_if_overwrite_true(self, model, model_csv_dir):
        model.to_csv(model_csv_dir, allow_overwrite=True)

    @pytest.mark.parametrize(
        "file_name",
        sorted(
            [
                f"inputs_{i}.csv"
                for i in calliope.examples.national_scale().inputs.data_vars.keys()
            ]
        ),
    )
    def test_save_csv(self, model_csv_dir, file_name):
        assert os.path.isfile(os.path.join(model_csv_dir, file_name))

    def test_csv_contents(self, model_csv_dir):
        with open(
            os.path.join(model_csv_dir, "inputs_flow_cap_max_systemwide.csv")
        ) as f:
            assert "demand_power" not in f.read()

    def test_save_csv_no_dropna(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            model.to_csv(out_path, dropna=False)

            with open(
                os.path.join(out_path, "inputs_flow_cap_max_systemwide.csv")
            ) as f:
                assert "demand_power" in f.read()

    @pytest.mark.xfail(reason="Not reimplemented the 'check feasibility' objective")
    def test_save_csv_not_optimal(self):
        model = calliope.examples.national_scale(
            scenario="check_feasibility",
            override_dict={"config.build.cyclic_storage": False},
        )

        model.build()
        model.solve()

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "out_dir")
            with pytest.warns(exceptions.ModelWarning):
                model.to_csv(out_path, dropna=False)

    @pytest.mark.parametrize("attr", ["config"])
    def test_dicts_as_model_attrs_and_property(self, model_from_file, attr):
        assert attr in model_from_file._model_data.attrs.keys()
        assert hasattr(model_from_file, attr)

    def test_defaults_as_model_attrs_not_property(self, model_from_file):
        assert "defaults" in model_from_file._model_data.attrs.keys()
        assert not hasattr(model_from_file, "defaults")

    @pytest.mark.parametrize("attr", ["results", "inputs"])
    def test_filtered_dataset_as_property(self, model_from_file, attr):
        assert hasattr(model_from_file, attr)

    def test_save_read_solve_save_netcdf(self, model, tmpdir_factory):
        out_path = tmpdir_factory.mktemp("model_dir").join("model.nc")
        model.to_netcdf(out_path)
        model_from_disk = calliope.read_netcdf(out_path)

        # Simulate a re-run via the backend
        model_from_disk.build()
        model_from_disk.solve(force=True)

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.nc")
            model_from_disk.to_netcdf(out_path)
            assert os.path.isfile(out_path)

    def test_save_lp(self, model):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model.lp")
            model.backend.to_lp(out_path)

            with open(out_path) as f:
                assert "variables(flow_cap)" in f.read()

    @pytest.mark.skip(
        reason="SPORES mode will fail until the cost max group constraint can be reproduced"
    )
    def test_save_per_spore(self):
        with tempfile.TemporaryDirectory() as tempdir:
            os.mkdir(os.path.join(tempdir, "output"))
            model = calliope.examples.national_scale(scenario="spores")
            model.build()
            model.solve(
                spores_save_per_spore=True,
                save_per_spore_path=os.path.join(tempdir, "output/spore_{}.nc"),
            )

            for i in ["0", "1", "2", "3"]:
                assert os.path.isfile(os.path.join(tempdir, "output", f"spore_{i}.nc"))
            assert not os.path.isfile(os.path.join(tempdir, "output.nc"))


class TestYaml:
    @pytest.fixture
    def dummy_yaml_import(self):
        return """
            import: ['somefile.yaml']
        """

    def test_do_not_resolve_imports(self, dummy_yaml_import):
        """Text inputs that attempt to import files should raise an error."""

        with pytest.raises(ValueError) as exinfo:  # noqa: PT011, false positive
            calliope.io.read_rich_yaml(dummy_yaml_import)

        assert check_error_or_warning(
            exinfo, "Imports are not possible for non-file yaml inputs."
        )

    @pytest.fixture
    def dummy_imported_file(self, tmp_path) -> Path:
        file = tmp_path / "test_import.yaml"
        text = """
            somekey.nested: 1
            anotherkey: 2
        """
        with open(file, "w") as f:
            f.write(text)
        return file

    def test_import(self, dummy_imported_file):
        file = dummy_imported_file.parent / "main_file.yaml"
        text = """
            import:
                - test_import.yaml
            foo:
                bar: 1
                baz: 2
                3:
                    4: 5
        """
        with open(file, "w") as f:
            f.write(text)
        d = calliope.io.read_rich_yaml(file)

        assert "somekey.nested" in d.keys_nested()
        assert d.get_key("anotherkey") == 2

    def test_import_must_be_list(self, tmp_path):
        file = tmp_path / "non_list_import.yaml"
        text = """
            import: test_import.yaml
            foo:
                bar: 1
                baz: 2
                3:
                    4: 5
        """
        with open(file, "w") as f:
            f.write(text)

        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            calliope.io.read_rich_yaml(file)
        assert check_error_or_warning(excinfo, "`import` must be a list.")

    def test_from_yaml_string(self):
        yaml_string = """
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
        d = calliope.io.read_rich_yaml(yaml_string)
        assert d.a == 1
        assert d.c.z.II == 2

    def test_from_yaml_string_dot_strings(self):
        yaml_string = "a.b.c: 1\na.b.foo: 2"
        d = calliope.io.read_rich_yaml(yaml_string)
        assert d.a.b.c == 1
        assert d.a.b.foo == 2

    def test_from_yaml_string_dot_strings_duplicate(self):
        yaml_string = "a.b.c: 1\na.b.c: 2"
        with pytest.raises(ruamel_yaml.constructor.DuplicateKeyError):
            calliope.io.read_rich_yaml(yaml_string)

    def test_simple_invalid_yaml(self):
        yaml_string = "1 this is not valid yaml"
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            calliope.io.read_rich_yaml(yaml_string)
        assert check_error_or_warning(excinfo, "Could not parse <yaml string> as YAML")

    def test_parser_error(self):
        with pytest.raises(ruamel_yaml.YAMLError):
            calliope.io.read_rich_yaml(
                """
            foo: bar
            baz: 1
                - foobar
                bar: baz

            """
            )

    @pytest.fixture
    def multi_order_yaml(self):
        return calliope.io.read_rich_yaml(
            """
            A.B.C: 10
            A.B:
                E: 20
            C: "foobar"
        """
        )

    def test_order_of_subdicts(self, multi_order_yaml):
        assert multi_order_yaml.A.B.C == 10
        assert multi_order_yaml.A.B.E == 20
        assert multi_order_yaml.C == "foobar"

    def test_as_dict_with_sublists(self):
        d = calliope.io.read_rich_yaml("a: [{x: 1}, {y: 2}]")
        dd = d.as_dict()
        assert dd["a"][0]["x"] == 1
        assert all(
            [isinstance(dd["a"][0], dict), not isinstance(dd["a"][0], AttrDict)]
        )  # Not AttrDict!

    def test_replacement_null_from_file(self, multi_order_yaml):
        replacement = calliope.io.read_rich_yaml("C._REPLACE_: null")
        multi_order_yaml.union(replacement, allow_override=True, allow_replacement=True)
        assert multi_order_yaml.C is None

    @pytest.fixture
    def yaml_from_path(self):
        this_path = Path(__file__).parent
        return calliope.io.read_rich_yaml(this_path / "common" / "yaml_file.yaml")

    def test_from_yaml_path(self, yaml_from_path):
        assert yaml_from_path.a == 1
        assert yaml_from_path.c.z.II == 2

    def test_to_yaml(self, yaml_from_path):
        yaml_from_path.set_key("numpy.some_int", np.int32(10))
        yaml_from_path.set_key("numpy.some_float", np.float64(0.5))
        yaml_from_path.a_list = [0, 1, 2]
        with tempfile.TemporaryDirectory() as tempdir:
            out_file = os.path.join(tempdir, "test.yaml")
            yaml_from_path.to_yaml(out_file)

            with open(out_file) as f:
                result = f.read()

            assert "some_int: 10" in result
            assert "some_float: 0.5" in result
            assert "a_list:\n- 0\n- 1\n- 2" in result


class TestYAMLTemplates:

    @pytest.fixture
    def dummy_solved_template(self) -> calliope.io.TemplateSolver:
        text = """
        templates:
            T1:
                A: ["foo", "bar"]
                B: 1
            T2:
                C: bar
                template: T1
            T3:
                template: T1
                B: 11
            T4:
                template: T3
                A: ["bar", "foobar"]
                B: "1"
                C: {"foo": "bar"}
                D: true
        a:
            template: T1
            a1: 1
        b:
            template: T3
        c:
            template: T4
            D: false
        """
        yaml_data = calliope.io.read_rich_yaml(text)
        return calliope.io.TemplateSolver(yaml_data)

    def test_inheritance_templates(self, dummy_solved_template):
        templates = dummy_solved_template.resolved_templates
        assert all(
            [
                templates.T1 == {"A": ["foo", "bar"], "B": 1},
                templates.T2 == {"A": ["foo", "bar"], "B": 1, "C": "bar"},
                templates.T3 == {"A": ["foo", "bar"], "B": 11},
                templates.T4 == {"A": ["bar", "foobar"], "B": "1", "C": {"foo": "bar"}, "D": True}
            ]
        )

    def test_template_inheritance_data(self, dummy_solved_template):
        data = dummy_solved_template.resolved_data
        assert all(
            [
                data.a == {"A": ["foo", "bar"], "B": 1, "a1": 1},
                data.b == {"A": ["foo", "bar"], "B": 11},
                data.c == {"A": ["bar", "foobar"], "B": "1", "C": {"foo": "bar"}, "D": False}
            ]
        )

    def test_invalid_template_error(self):
        text = calliope.io.read_rich_yaml(
            """
            templates:
                T1: "not_a_yaml_block"
                T2:
                    foo: bar
            a:
                template: T2
            """
        )
        with pytest.raises(ValueError, match="Template definitions must be YAML blocks."):
            calliope.io.TemplateSolver(text)

    def test_circular_template_error(self):
        text = calliope.io.read_rich_yaml(
            """
            templates:
                T1:
                    template: T2
                    bar: foo
                T2:
                    template: T1
                    foo: bar
            a:
                template: T2
            """
        )
        with pytest.raises(ValueError, match="Circular template reference detected"):
            calliope.io.TemplateSolver(text)

    def test_incorrect_template_placement_error(self):
        text = calliope.io.read_rich_yaml(
            """
            templates:
                T1:
                    stuff: null
                T2:
                    foo: bar
            a:
                template: T2
            b:
                templates:
                    T3:
                        this: "should not be here"
            """
        )
        with pytest.raises(ValueError, match="Template definitions must be placed at the top level of the YAML file."):
            calliope.io.TemplateSolver(text)


