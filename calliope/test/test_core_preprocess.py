
import os
import calliope
import pytest  # pylint: disable=unused-import

from calliope.core.attrdict import AttrDict
import calliope.exceptions as exceptions
import pandas as pd


this_path = os.path.dirname(__file__)
model_location = os.path.join(this_path, 'common', 'test_model', 'model.yaml')
override_location = os.path.join(this_path, 'common', 'test_model', 'overrides.yaml')


def run_model(override_dict, override_groups):
    return calliope.Model(
        model_location, override_dict=override_dict,
        override_file=override_location + ':' + override_groups
    )


class TestModelRun:
    def test_override_base_technology_groups(self):
        """
        Test that base technology are not being overriden by any user input
        """
        override = AttrDict.from_yaml_string(
            """
            tech_groups:
                supply:
                    constraints:
                        energy_cap_max: 1000
            """
        )
        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override, override_groups='simple_supply,one_day')

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
            locations:
                1:
                    techs:
                        test_undefined_carrier:
            """
        )
        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_incorrect_subset_time(self):
        """
        If subset_time is a list, it must have two entries (start_time, end_time)
        If subset_time is not a list, it should successfully subset on the given
        string/integer
        """
        # should fail: one string in list
        override1 = AttrDict.from_yaml_string(
            """
            model.subset_time: ['2005-01']
            """
        )
        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override1, override_groups='simple_supply')

        # should fail: three strings in list
        override2 = AttrDict.from_yaml_string(
            """
            model.subset_time: ['2005-01-01', '2005-01-02', '2005-01-03']
            """
        )
        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override2, override_groups='simple_supply')

        # should pass: two string in list as slice
        override3 = AttrDict.from_yaml_string(
            """
            model.subset_time: ['2005-01-01', '2005-01-07']
            """
        )
        model = run_model(override_dict=override3, override_groups='simple_supply')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-01-07 23:00:00', freq='H'))

        # should pass: one integer/string
        override3 = AttrDict.from_yaml_string(
            """
            model.subset_time: 2005-01
            """
        )
        model = run_model(override_dict=override3, override_groups='simple_supply')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-01-31 23:00:00', freq='H'))

        # should fail: time subset out of range of input data
        override3 = AttrDict.from_yaml_string(
            """
            model.subset_time: 2005-03
            """
        )
        with pytest.raises(KeyError):
            run_model(override_dict=override3, override_groups='simple_supply')

        # should fail: time subset out of range of input data
        override3 = AttrDict.from_yaml_string(
            """
            model.subset_time: ['2005-02-01', '2005-02-05']
            """
        )
        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override3, override_groups='simple_supply')

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
        model = run_model(override_dict=override1, override_groups='simple_conversion')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-02-01 23:00:00', freq='H'))

        # should fail: wrong dateformat input for one file
        override2 = AttrDict.from_yaml_string(
            """
            techs.test_demand_heat.constraints.resource: file=demand_heat_diff_dateformat.csv
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override2, override_groups='simple_conversion')


        # should fail: wrong dateformat input for all files
        override3 = AttrDict.from_yaml_string(
            """
            model.timeseries_dateformat: "%d/%m/%Y %H:%M:%S"
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override3, override_groups='simple_supply')

        # should fail: one value wrong in file
        override1 = AttrDict.from_yaml_string(
            """
            techs.test_demand_heat.constraints.resource: file=demand_heat_wrong_dateformat.csv
            """
        )
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.raises(exceptions.ModelError):
            model = run_model(override_dict=override1, override_groups='simple_conversion')

    def test_inconsistent_time_indeces(self):
        """
        Test that, including after any time subsetting, the indeces of all time
        varying input data are consistent with each other
        """

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
            locations.0.test_supply.constraints.resource: 10
            locations.0,1.test_supply.constraints.resource: 15
            """
        )

        with pytest.raises(KeyError):
            run_model(override_dict=override, override_groups='simple_supply,one_day')


class TestChecks:
    def test_unknown_carrier_tier(self):
        """
        User can only use 'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3',
        'out_3', 'ratios']
        """
        override1 = AttrDict.from_yaml_string(
            """
            techs.test_supply.essentials.carrier_1: power
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override1, override_groups='simple_supply,one_day')

        override2 = AttrDict.from_yaml_string(
            """
            techs.test_supply.essentials.carrier_out_4: power
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override2, override_groups='simple_supply,one_day')

    def test_name_overlap(self):
        """
        No tech_groups/techs may have the same identifier as the built-in groups
        """
        override1 = AttrDict.from_yaml_string(
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
                1:
                    techs:
                        supply:
                0:
                    techs:
                        supply:
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override1, override_groups='one_day')

        override2 = AttrDict.from_yaml_string(
            """
            tech_groups:
                supply:
                    essentials:
                        name: Supply tech
                        carrier: gas
                        parent: supply_plus
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override2, override_groups='simple_supply,one_day')

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
            run_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_resource_as_carrier(self):
        """
        No carrier in technology or technology group can be called `resource`
        """

        override1 = AttrDict.from_yaml_string(
            """
            techs:
                test_supply:
                    essentials:
                        name: Supply tech
                        carrier: resource
                        parent: supply
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override1, override_groups='simple_supply,one_day')

        override2 = AttrDict.from_yaml_string(
            """
            tech_groups:
                test_supply_group:
                    essentials:
                        name: Supply tech
                        carrier: resource
                        parent: supply
            techs:
                test_supply:
                    essentials:
                        parent: test_supply_group
            """
        )

        with pytest.raises(exceptions.ModelError):
            run_model(override_dict=override2, override_groups='simple_supply,one_day')

    def test_missing_required_constraints(self):
        """
        A technology within an abstract base technology must define a subset of
        hardcoded constraints in order to function
        """

    def test_defining_non_allowed_constraints(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded constraints, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """

    def test_defining_non_allowed_costs(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded costs, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """

    def test_exporting_unspecified_carrier(self):
        """
        User can only define an export carrier if it is defined in
        ['carrier_out', 'carrier_out_2', 'carrier_out_3']
        """

    def test_allowed_time_varying_constraints(self):
        """
        `file=` is only allowed on a hardcoded list of constraints, unless
        `_time_varying` is appended to the constraint (i.e. user input)
        """

    def test_incorrect_location_coordinates(self):
        """
        Either all or no locations must have `coordinates` defined and, if all
        defined, they must be in the same coordinate system (lat/lon or x/y)
        """


class TestDataset:
    def test_inconsistent_timesteps(self):
        """
        Timesteps must be consistent?
        """

    def test_unassigned_sets(self):
        """
        Check that all sets in which there are possible loc:techs are assigned
        and have been filled
        """

    def test_negative_cost_unassigned_cap(self):
        """
        Any negative cost associated with a capacity (e.g. cost_energy_cap) must
        be applied to a capacity iff the upper bound of that capacity has been defined
        """

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
