
import os
import calliope
import pytest  # pylint: disable=unused-import

from calliope.core.attrdict import AttrDict
import calliope.exceptions as exceptions
import pandas as pd


this_path = os.path.dirname(__file__)
model_location = os.path.join(this_path, 'common', 'test_model', 'model.yaml')
override_location = os.path.join(this_path, 'common', 'test_model', 'overrides.yaml')

constraint_sets = AttrDict.from_yaml(os.path.join(this_path, 'common', 'constraint_sets.yaml'))

_defaults_files = {
    k: os.path.join(os.path.dirname(calliope.__file__), 'config', k + '.yaml')
    for k in ['model', 'defaults']
}
defaults = AttrDict.from_yaml(_defaults_files['defaults'])
defaults_model = AttrDict.from_yaml(_defaults_files['model'])


def build_model(override_dict, override_groups):
    return calliope.Model(
        model_location, override_dict=override_dict,
        override_file=override_location + ':' + override_groups
    )


class TestModelRun:
    def test_undefined_carriers(self):
        """
        Test that user has input either carrier or carrier_in/_out for each tech
        """
        override = AttrDict.from_yaml_string(
            """
            techs:
                test_undefined_carrier:
                    essentials:
                        parent: supply
                        name: test
                    constraints:
                        resource: .inf
                        energy_cap_max: .inf
            locations.1.techs.test_undefined_carrier:
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_incorrect_subset_time(self):
        """
        If subset_time is a list, it must have two entries (start_time, end_time)
        If subset_time is not a list, it should successfully subset on the given
        string/integer
        """

        override = lambda param: AttrDict.from_yaml_string(
            "model.subset_time: {}".format(param)
        )

        # should fail: one string in list
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override(['2005-01']), override_groups='simple_supply')

        # should fail: three strings in list
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override(['2005-01-01', '2005-01-02', '2005-01-03']), override_groups='simple_supply')

        # should pass: two string in list as slice
        model = build_model(override_dict=override(['2005-01-01', '2005-01-07']), override_groups='simple_supply')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-01-07 23:00:00', freq='H'))

        # should pass: one integer/string
        model = build_model(override_dict=override('2005-01'), override_groups='simple_supply')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-01-31 23:00:00', freq='H'))

        # should fail: time subset out of range of input data
        with pytest.raises(KeyError):
            build_model(override_dict=override('2005-03'), override_groups='simple_supply')

        # should fail: time subset out of range of input data
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override(['2005-02-01', '2005-02-05']), override_groups='simple_supply')

    def test_incorrect_date_format(self):
        """
        Test the date parser catches a different date format from file than
        user input/default (inc. if it is just one line of a file that is incorrect)
        """

        # should pass: changing datetime format from default
        override1 = AttrDict.from_yaml_string(
            """
            model.timeseries_dateformat: "%d/%m/%Y %H:%M:%S"
            techs.test_demand_heat.constraints.resource: file=demand_heat_diff_dateformat.csv
            techs.test_demand_elec.constraints.resource: file=demand_heat_diff_dateformat.csv
            """
        )
        model = build_model(override_dict=override1, override_groups='simple_conversion')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-02-01 23:00:00', freq='H'))

        # should fail: wrong dateformat input for one file
        override2 = AttrDict.from_yaml_string(
            """
            techs.test_demand_heat.constraints.resource: file=demand_heat_diff_dateformat.csv
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override2, override_groups='simple_conversion')

        # should fail: wrong dateformat input for all files
        override3 = AttrDict.from_yaml_string(
            """
            model.timeseries_dateformat: "%d/%m/%Y %H:%M:%S"
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override3, override_groups='simple_supply')

        # should fail: one value wrong in file
        override4 = AttrDict.from_yaml_string(
            """
            techs.test_demand_heat.constraints.resource: file=demand_heat_wrong_dateformat.csv
            """
        )
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override4, override_groups='simple_conversion')

    def test_inconsistent_time_indeces(self):
        """
        Test that, including after any time subsetting, the indeces of all time
        varying input data are consistent with each other
        """
        # should fail: wrong length of demand_heat csv vs demand_elec
        override1 = AttrDict.from_yaml_string(
            """
            techs.test_demand_heat.constraints.resource: file=demand_heat_wrong_length.csv
            """
        )
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, override_groups='simple_conversion')

        # should pass: wrong length of demand_heat csv, but time subsetting removes the difference
        build_model(override_dict=override1, override_groups='simple_conversion,one_day')

    def test_empty_key_on_explode(self):
        """
        On exploding locations (from ``'1--3'`` or ``'1,2,3'`` to
        ``['1', '2', '3']``), raise error on the resulting list being empty
        """
        list1 = calliope.core.preprocess.locations.explode_locations('1--3')
        list2 = calliope.core.preprocess.locations.explode_locations('1,2,3')

        assert list1 == list2 == ['1', '2', '3']

    def test_key_clash_on_set_loc_key(self):
        """
        Raise error on attempted overwrite of information regarding a recently
        exploded location
        """
        override = AttrDict.from_yaml_string(
            """
            locations.0.test_supply_elec.constraints.resource: 10
            locations.0,1.test_supply_elec.constraints.resource: 15
            """
        )

        with pytest.raises(KeyError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')


class TestChecks:
    def test_model_version_mismatch(self):
        """
        Model config says model.calliope_version = 0.1, which is not what we
        are running, so we want a warning.
        """
        override = AttrDict.from_yaml_string(
            """
            model.calliope_version: 0.1
            """
        )

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, override_groups='simple_supply,one_day')

        all_warnings = ','.join(str(excinfo.list[i]) for i in range(len(excinfo.list)))

        assert 'Model configuration specifies calliope_version' in all_warnings

    def test_unknown_carrier_tier(self):
        """
        User can only use 'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3',
        'out_3', 'ratios']
        """

        override1 = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.essentials.carrier_1: power
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, override_groups='simple_supply,one_day')

        override2 = AttrDict.from_yaml_string(
            """
            techs.test_conversion_plus.essentials.carrier_out_4: power
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override2, override_groups='simple_conversion_plus,one_day')

    def test_name_overlap(self):
        """
        No tech may have the same identifier as a tech group
        """
        override = AttrDict.from_yaml_string(
            """
            techs:
                supply:
                    essentials:
                        name: Supply tech
                        carrier: gas
                        parent: supply
                    constraints:
                        energy_cap_max: 10
                        resource: .inf
            locations:
                1.techs.supply:
                0.techs.supply:
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='one_day')

    def test_unspecified_parent(self):
        """
        All technologies and technology groups must specify a parent
        """

        override = AttrDict.from_yaml_string(
            """
            techs.test_supply_no_parent:
                    essentials:
                        name: Supply tech
                        carrier: gas
                    constraints:
                        energy_cap_max: 10
                        resource: .inf
            locations.1.test_supply_no_parent:
            """
        )

        with pytest.raises(KeyError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_resource_as_carrier(self):
        """
        No carrier in technology or technology group can be called `resource`
        """

        override1 = AttrDict.from_yaml_string(
            """
            techs:
                test_supply_elec:
                    essentials:
                        name: Supply tech
                        carrier: resource
                        parent: supply
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, override_groups='simple_supply,one_day')

        override2 = AttrDict.from_yaml_string(
            """
            tech_groups:
                test_supply_group:
                    essentials:
                        name: Supply tech
                        carrier: resource
                        parent: supply
            techs.test_supply_elec.essentials.parent: test_supply_group
            """
        )

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override2, override_groups='simple_supply,one_day')

    def test_missing_constraints(self):
        """
        A technology must define at least one constraint.
        """

        override = AttrDict.from_yaml_string(
            """
            techs:
                supply_missing_constraint:
                    essentials:
                        parent: supply
                        carrier: electricity
                        name: supply missing constraint
            locations.1.techs.supply_missing_constraint:
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_missing_required_constraints(self):
        """
        A technology within an abstract base technology must define a subset of
        hardcoded constraints in order to function
        """
        # should fail: missing one of ['energy_cap_max', 'energy_cap_equals', 'energy_cap_per_unit']
        override_supply1 = AttrDict.from_yaml_string(
            """
            techs:
                supply_missing_constraint:
                    essentials:
                        parent: supply
                        carrier: electricity
                        name: supply missing constraint
                    constraints:
                        resource_area_max: 10
            locations.1.techs.supply_missing_constraint:
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override_supply1, override_groups='simple_supply,one_day')

        # should pass: giving one of ['energy_cap_max', 'energy_cap_equals', 'energy_cap_per_unit']
        override_supply2 = AttrDict.from_yaml_string(
            """
            techs:
                supply_missing_constraint:
                    essentials:
                        parent: supply
                        carrier: electricity
                        name: supply missing constraint
                    constraints.energy_cap_max: 10
            locations.1.techs.supply_missing_constraint:
            """
        )
        build_model(override_dict=override_supply2, override_groups='simple_supply,one_day')

    def test_defining_non_allowed_constraints(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded constraints, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """
        # should fail: storage_cap_max not allowed for supply tech
        override_supply1 = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.constraints.storage_cap_max: 10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override_supply1, override_groups='simple_supply,one_day')

    def test_defining_non_allowed_costs(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded costs, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """
        # should fail: storage_cap_max not allowed for supply tech
        override = AttrDict.from_yaml_string(
            """
            techs.test_supply_elec.costs.monetary.storage_cap: 10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

        # should fail: om_prod not allowed for demand tech
        override = AttrDict.from_yaml_string(
            """
            techs.test_demand_elec.costs.monetary.om_prod: 10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_exporting_unspecified_carrier(self):
        """
        User can only define an export carrier if it is defined in
        ['carrier_out', 'carrier_out_2', 'carrier_out_3']
        """
        override_supply = lambda param: AttrDict.from_yaml_string(
            "techs.test_supply_elec.constraints.export_carrier: {}".format(param)
        )

        override_converison_plus = lambda param: AttrDict.from_yaml_string(
            "techs.test_conversion_plus.constraints.export_carrier: {}".format(param)
        )

        # should fail: exporting `heat` not allowed for electricity supply tech
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override_supply('heat'), override_groups='simple_supply,one_day')

        # should fail: exporting `random` not allowed for conversion_plus tech
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override_converison_plus('random'), override_groups='simple_conversion_plus,one_day')

        # should pass: exporting electricity for supply tech
        build_model(override_dict=override_supply('electricity'), override_groups='simple_supply,one_day')

        # should pass: exporting heat for conversion tech
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override_converison_plus('heat'), override_groups='simple_conversion_plus,one_day')
        all_warnings = ','.join(str(excinfo.list[i]) for i in range(len(excinfo.list)))
        assert (
            'dimension loc_techs_transmission and associated variables distance, '
            'lookup_remotes were empty, so have been deleted' in all_warnings
        )

    def test_allowed_time_varying_constraints_supply(self):
        """
        `file=` is only allowed on a hardcoded list of constraints, unless
        `_time_varying` is appended to the constraint (i.e. user input)
        """

        allowed_constraints_no_file = list(
            set(defaults_model.tech_groups.supply.allowed_constraints)
            .difference(defaults.file_allowed)
        )

        allowed_constraints_file = list(
            set(defaults_model.tech_groups.supply.allowed_constraints)
            .intersection(defaults.file_allowed)
        )

        override = lambda param: AttrDict.from_yaml_string(
            "techs.test_supply_elec.constraints.{}: file=binary_one_day.csv".format(param)
        )

        # should fail: Cannot have `file=` on the following constraints
        for param in allowed_constraints_no_file:
            with pytest.raises(exceptions.ModelError):
                build_model(override_dict=override(param), override_groups='simple_supply,one_day')

        # should pass: can have `file=` on the following constraints
        for param in allowed_constraints_file:
            build_model(override_dict=override(param), override_groups='simple_supply,one_day')

    def test_allowed_time_varying_constraints_storage(self):
        """
        `file=` is only allowed on a hardcoded list of constraints, unless
        `_time_varying` is appended to the constraint (i.e. user input)
        """

        allowed_constraints_no_file = list(
            set(defaults_model.tech_groups.storage.allowed_constraints)
            .difference(defaults.file_allowed)
        )

        allowed_constraints_file = list(
            set(defaults_model.tech_groups.storage.allowed_constraints)
            .intersection(defaults.file_allowed)
        )

        override = lambda param: AttrDict.from_yaml_string(
            "techs.test_storage.constraints.{}: file=binary_one_day.csv".format(param)
        )

        # should fail: Cannot have `file=` on the following constraints
        for param in allowed_constraints_no_file:
            with pytest.raises(exceptions.ModelError):
                build_model(override_dict=override(param), override_groups='simple_storage,one_day')

        # should pass: can have `file=` on the following constraints
        for param in allowed_constraints_file:
            build_model(override_dict=override(param), override_groups='simple_storage,one_day')

    def test_incorrect_location_coordinates(self):
        """
        Either all or no locations must have `coordinates` defined and, if all
        defined, they must be in the same coordinate system (lat/lon or x/y)
        """

        override = lambda param0, param1: AttrDict.from_yaml_string(
            """
            locations:
                0.coordinates: {}
                1.coordinates: {}
            """.format(param0, param1)
        )
        cartesian0 = {'x': 0, 'y': 1}
        cartesian1 = {'x': 1, 'y': 1}
        geographic0 = {'lat': 0, 'lon': 1}
        geographic1 = {'lat': 1, 'lon': 1}

        # should fail: cannot have locations in one place and not in another
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override(cartesian0, 'null'), override_groups='simple_storage,one_day')

        # should fail: cannot have cartesian coordinates in one place and geographic in another
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override(cartesian0, geographic1), override_groups='simple_storage,one_day')

        # should pass: cartesian coordinates in both places
        build_model(override_dict=override(cartesian0, cartesian1), override_groups='simple_storage,one_day')

        # should pass: geographic coordinates in both places
        build_model(override_dict=override(geographic0, geographic1), override_groups='simple_storage,one_day')

    def test_milp_constraints(self):
        """
        If `units` is defined, but not `energy_cap_per_unit`, throw an error
        """

        # should fail: no energy_cap_per_unit
        override1 = AttrDict.from_yaml_string("techs.test_supply_elec.constraints.units_max: 4")

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override1, override_groups='simple_supply,one_day')

        # should pass: energy_cap_per_unit given
        override2 = AttrDict.from_yaml_string("""
            techs.test_supply_elec.constraints:
                        units_max: 4
                        energy_cap_per_unit: 5
            """)

        build_model(override_dict=override2, override_groups='simple_supply,one_day')


class TestDataset:

    # FIXME: What are we testing here?
    def test_inconsistent_timesteps(self):
        """
        Timesteps must be consistent?
        """

    def test_unassigned_sets(self):
        """
        Check that all sets in which there are possible loc:techs are assigned
        and have been filled
        """
        models = dict()
        models['model_national'] = calliope.examples.national_scale()
        models['model_urban'] = calliope.examples.urban_scale()
        models['model_milp'] = calliope.examples.milp()

        for model_name, model in models.items():
            for set_name, set_vals in model._model_data.coords.items():
                if 'constraint' in set_name:
                    assert set(set_vals.values) == set(constraint_sets[model_name][set_name])

    def test_negative_cost_unassigned_cap(self):
        """
        Any negative cost associated with a capacity (e.g. cost_energy_cap) must
        be applied to a capacity iff the upper bound of that capacity has been defined
        """

        # should fail: resource_cap cost is negtive, resource_cap_max is infinite
        override = AttrDict.from_yaml_string(
            "techs.test_supply_plus.costs.monetary.resource_cap: -10"
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_supply_plus,one_day')

        # should fail: storage_cap cost is negative, storage_cap_max is infinite
        override = AttrDict.from_yaml_string(
            """
            techs.test_storage:
                    constraints.storage_cap_max: .inf
                    costs.monetary.storage_cap: -10
            """
        )
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_storage,one_day')

    def test_missing_array(self):
        """
        Check that the dataset includes all arrays *required* for a model to function
        """

    def test_missing_attrs(self):
        """
        Check that the dataset includes all attributes *required* for a model to function
        """


class TestUtil():
    def test_concat_iterable_ensures_same_length_iterables(self):
        """
        All iterables must have the same length
        """

    def test_concat_iterable_check_concatenators(self):
        """
        Contatenators should be one shorter than the length of each iterable
        """
