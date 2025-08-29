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
    def vars_to_add_attrs(self) -> dict:
        return {"inputs": "source_use_max", "results": "flow_cap"}

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
        model.build()
        model.solve()

        model.inputs.attrs = attrs

        for ds, var in vars_to_add_attrs.items():
            getattr(model, ds)[var] = getattr(model, ds)[var].assign_attrs(**attrs)

        return model

    @pytest.fixture(scope="class")
    def model_file(self, tmpdir_factory, model):
        out_path = tmpdir_factory.mktemp("data").join("model.nc")
        model.to_netcdf(out_path)
        return out_path

    @pytest.fixture(scope="class")
    def model_from_file(self, model_file):
        return calliope.read_netcdf(model_file)

    @pytest.fixture(scope="class")
    def model_from_file_no_processing(self, model_file):
        return xr.open_dataset(model_file, group="inputs")

    @pytest.fixture(scope="class")
    def model_csv_dir(self, tmpdir_factory, model):
        parent_path = tmpdir_factory.mktemp("data")
        out_path = parent_path / "csvs"
        model.to_csv(out_path)
        return out_path

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
        var_attrs = [
            getattr(model, ds)[var].attrs for ds, var in vars_to_add_attrs.items()
        ]
        for attrs in [model.inputs.attrs, *var_attrs]:
            assert isinstance(attrs[attr], expected_type)
            if expected_val is None:
                assert attrs[attr] is None
            else:
                assert attrs[attr] == expected_val

    @pytest.mark.parametrize(
        ("serialisation_list_name", "list_elements"),
        [
            ("serialised_bools", ["foo_true", "foo_false"]),
            ("serialised_nones", ["foo_none"]),
            ("serialised_dicts", ["foo_dict", "foo_attrdict"]),
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

    def test_save_csv_dir_can_exist_if_overwrite_true(
        self, model: calliope.Model, model_csv_dir
    ):
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

    def test_config_reload(self, model_from_file, model):
        assert model_from_file.config.model_dump() == model.config.model_dump()

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
    TEST_TEXT = {
        "simple_nested": """
somekey.nested: 1
anotherkey: 2
""",
        "triple_nested": """
foo:
    bar: 1
    baz: 2
    nested:
        value: 5
""",
        "complex_commented": """
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
e: False
""",
        "nested_string": "a.b.c: 1\na.b.foo: 2\nb.a.c.bar: foo",
    }

    TEST_EXPECTED = {
        "simple_nested": {"somekey": {"nested": 1}, "anotherkey": 2},
        "triple_nested": {"foo": {"bar": 1, "baz": 2, "nested": {"value": 5}}},
        "complex_commented": {
            "a": 1,
            "b": 2,
            "c": {"x": "foo", "y": "bar", "z": {"I": 1, "II": 2}},
            "d": None,
            "e": False,
        },
        "nested_string": {
            "a": {"b": {"c": 1, "foo": 2}},
            "b": {"a": {"c": {"bar": "foo"}}},
        },
    }

    @pytest.fixture(
        params=["simple_nested", "triple_nested", "complex_commented", "nested_string"]
    )
    def test_group(self, request) -> str:
        return request.param

    @pytest.fixture
    def yaml_text(self, test_group) -> str:
        return self.TEST_TEXT[test_group]

    @pytest.fixture
    def expected_dict(self, test_group) -> dict:
        return self.TEST_EXPECTED[test_group]

    @pytest.fixture
    def dummy_imported_file(self, tmp_path) -> Path:
        file = tmp_path / "test_import.yaml"
        text = """
# Comment
import_key_a.nested: 1
import_key_b: 2
import_key_c: [1, 2, 3]
        """
        file.write_text(text)
        return file

    def test_text_read(self, yaml_text, expected_dict):
        """Loading from text strings should be correct."""
        read = calliope.io.read_rich_yaml(yaml_text)
        assert read == expected_dict

    def test_file_read(self, test_group, yaml_text, expected_dict, tmp_path):
        """Loading from files should be correct."""
        file = tmp_path / f"{test_group}.yaml"
        file.write_text(yaml_text)
        read = calliope.io.read_rich_yaml(file)
        assert read == expected_dict

    @pytest.mark.parametrize(
        "bad_import",
        [
            "import: ['somefile.yaml']\n",
            "import: ['somefile.yaml', 'other_file.yaml']\n",
        ],
    )
    def test_text_import_error(self, yaml_text, bad_import):
        """Text inputs that attempt to import files should raise an error."""
        with pytest.raises(
            ValueError, match="Imports are not possible for non-file yaml inputs."
        ):
            calliope.io.read_rich_yaml(bad_import + yaml_text)

    def test_import(self, test_group, yaml_text, dummy_imported_file):
        """Imported files relative to the main file should load correctly."""
        file = dummy_imported_file.parent / f"{test_group}_relative.yaml"
        import_text = f"""
import:
    - {dummy_imported_file.name}
"""
        file.write_text(import_text + yaml_text)
        d = calliope.io.read_rich_yaml(file)

        assert "import_key_a.nested" in d.keys_nested()
        assert d.get_key("import_key_b") == 2
        assert d["import_key_c"] == [1, 2, 3]

    def test_invalid_import_type_error(
        self, test_group, yaml_text, dummy_imported_file
    ):
        file = dummy_imported_file.parent / f"{test_group}_invalid_import_type.yaml"
        import_text = f"""import: {dummy_imported_file.name}\n"""
        file.write_text(import_text + yaml_text)

        with pytest.raises(ValueError) as excinfo:  # noqa: PT011, false positive
            calliope.io.read_rich_yaml(file)
        assert check_error_or_warning(excinfo, "`import` must be a list.")

    def test_duplicate_dot_string_error(self):
        """Duplicate entries should result in an error."""
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

    def test_as_dict_with_sublists(self):
        """Lists should not be converted to AttrDict."""
        d = calliope.io.read_rich_yaml("a: [{x: 1}, {y: 2}]")
        dd = d.as_dict()
        assert dd["a"][0]["x"] == 1
        assert all([isinstance(dd["a"][0], dict), not isinstance(dd["a"][0], AttrDict)])

    def test_replacement_null_from_file(self):
        yaml_dict = calliope.io.read_rich_yaml(
            """
            A.B.C: 10
            A.B:
                E: 20
            C: "foobar"
        """
        )
        replacement = calliope.io.read_rich_yaml("C._REPLACE_: null")
        yaml_dict.union(replacement, allow_override=True, allow_replacement=True)
        assert yaml_dict.C is None

    def test_to_yaml_roundtrip(self, expected_dict):
        """Saving to a file should result in no data loss."""
        yaml_text = calliope.io.to_yaml(expected_dict)
        reloaded = calliope.io.read_rich_yaml(yaml_text)
        assert reloaded == expected_dict

    def test_to_yaml_complex(self, yaml_text):
        """Saving to a file/string should handle special cases."""
        yaml_dict = calliope.io.read_rich_yaml(yaml_text)
        yaml_dict.set_key("numpy.some_int", np.int32(10))
        yaml_dict.set_key("numpy.some_float", np.float64(0.5))
        yaml_dict.a_list = [0, 1, 2]
        with tempfile.TemporaryDirectory() as tempdir:
            out_file = Path(tempdir) / "test.yaml"
            calliope.io.to_yaml(yaml_dict, path=out_file)

            result = out_file.read_text()

            assert "some_int: 10" in result
            assert "some_float: 0.5" in result
            assert "a_list:\n- 0\n- 1\n- 2" in result
