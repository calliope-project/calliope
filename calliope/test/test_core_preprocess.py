import pytest  # pylint: disable=unused-import
import os

import pandas as pd
import numpy as np

import calliope
import calliope.exceptions as exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.preprocess import time

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import \
    constraint_sets, defaults, defaults_model, check_error_or_warning


class TestModelRun:
    def test_model_from_dict(self):
        """
        Test loading a file from dict/AttrDict instead of from YAML
        """
        this_path = os.path.dirname(__file__)
        model_location = os.path.join(this_path, 'common', 'test_model', 'model.yaml')
        model_dict = AttrDict.from_yaml(model_location)
        location_dict = AttrDict({
            'locations': {
                '0': {'techs': {'test_supply_elec': {}, 'test_demand_elec': {}}},
                '1': {'techs': {'test_supply_elec': {}, 'test_demand_elec': {}}}
            }
        })
        model_dict.union(location_dict)
        model_dict.model['timeseries_data_path'] = os.path.join(
            this_path, 'common', 'test_model', model_dict.model['timeseries_data_path']
        )
        # test as AttrDict
        calliope.Model(model_dict)

        # test as dict
        calliope.Model(model_dict.as_dict())

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
        override1 = {
            'model.timeseries_dateformat': "%d/%m/%Y %H:%M:%S",
            'techs.test_demand_heat.constraints.resource': 'file=demand_heat_diff_dateformat.csv',
            'techs.test_demand_elec.constraints.resource': 'file=demand_heat_diff_dateformat.csv'
        }
        model = build_model(override_dict=override1, override_groups='simple_conversion')
        assert all(model.inputs.timesteps.to_index() == pd.date_range('2005-01', '2005-02-01 23:00:00', freq='H'))

        # should fail: wrong dateformat input for one file
        override2 = {
            'techs.test_demand_heat.constraints.resource': 'file=demand_heat_diff_dateformat.csv'
        }

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override2, override_groups='simple_conversion')

        # should fail: wrong dateformat input for all files
        override3 = {
            'model.timeseries_dateformat': "%d/%m/%Y %H:%M:%S"
        }

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override3, override_groups='simple_supply')

        # should fail: one value wrong in file
        override4 = {
            'techs.test_demand_heat.constraints.resource': 'file=demand_heat_wrong_dateformat.csv'
        }
        # check in output error that it points to: 07/01/2005 10:00:00
        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override4, override_groups='simple_conversion')

    def test_inconsistent_time_indeces(self):
        """
        Test that, including after any time subsetting, the indeces of all time
        varying input data are consistent with each other
        """
        # should fail: wrong length of demand_heat csv vs demand_elec
        override1 = {
            'techs.test_demand_heat.constraints.resource': 'file=demand_heat_wrong_length.csv'
        }
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
        override = {
            'locations.0.test_supply_elec.constraints.resource': 10,
            'locations.0,1.test_supply_elec.constraints.resource': 15
        }

        with pytest.raises(KeyError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_calculate_depreciation(self):
        """
        Technologies which define investment costs *must* define lifetime and
        interest rate, so that a depreciation rate can be calculated.
        If lifetime == inf and interested > 0, depreciation rate will be inf, so
        we want to avoid that too.
        """

        override1 = {
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override1, override_groups='simple_supply,one_day')
        assert check_error_or_warning(
            error, 'Must specify constraints.lifetime and costs.monetary.interest_rate'
        )

        override2 = {
            'techs.test_supply_elec.constraints.lifetime': 10,
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override2, override_groups='simple_supply,one_day')
        assert check_error_or_warning(
            error, 'Must specify constraints.lifetime and costs.monetary.interest_rate'
        )

        override3 = {
            'techs.test_supply_elec.costs.monetary.interest_rate': 0.1,
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override3, override_groups='simple_supply,one_day')
        assert check_error_or_warning(
            error, 'Must specify constraints.lifetime and costs.monetary.interest_rate'
        )

        override4 = {
            'techs.test_supply_elec.constraints.lifetime': 10,
            'techs.test_supply_elec.costs.monetary.interest_rate': 0,
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override4, override_groups='simple_supply,one_day')
        assert check_error_or_warning(excinfo, '`monetary` interest rate of zero')

        override5 = {
            'techs.test_supply_elec.constraints.lifetime': np.inf,
            'techs.test_supply_elec.costs.monetary.interest_rate': 0,
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override5, override_groups='simple_supply,one_day')
        assert check_error_or_warning(
            excinfo, 'No investment monetary cost will be incurred for `test_supply_elec`'
        )

        override6 = {
            'techs.test_supply_elec.constraints.lifetime': np.inf,
            'techs.test_supply_elec.costs.monetary.interest_rate': 0.1,
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override6, override_groups='simple_supply,one_day')
        assert check_error_or_warning(
            excinfo, 'No investment monetary cost will be incurred for `test_supply_elec`'
        )

        override7 = {
            'techs.test_supply_elec.constraints.lifetime': 10,
            'techs.test_supply_elec.costs.monetary.interest_rate': 0.1,
            'techs.test_supply_elec.costs.monetary.energy_cap': 10
        }
        build_model(override_dict=override7, override_groups='simple_supply,one_day')


class TestChecks:

    def test_unrecognised_config_keys(self):
        """
        Check that the only top level keys can be 'model', 'run', 'locations',
        'techs', 'tech_groups' (+ 'config_path', but that is an internal addition)
        """
        override = {'nonsensical_key': 'random_string'}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, override_groups='simple_supply')

        assert check_error_or_warning(
            excinfo, 'Unrecognised top-level configuration item: nonsensical_key'
        )

    def test_unrecognised_model_run_keys(self):
        """
        Check that the only keys allowed in 'model' and 'run' are those in the
        model defaults
        """
        override1 = {'model.nonsensical_key': 'random_string'}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override1, override_groups='simple_supply')

        assert check_error_or_warning(
            excinfo, 'Unrecognised setting in model configuration: nonsensical_key'
        )

        override2 = {'run.nonsensical_key': 'random_string'}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override2, override_groups='simple_supply')

        assert check_error_or_warning(
            excinfo, 'Unrecognised setting in run configuration: nonsensical_key'
        )

        # A key that should be in run but is given in model
        override3 = {'model.solver': 'glpk'}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override3, override_groups='simple_supply')

        assert check_error_or_warning(
            excinfo, 'Unrecognised setting in model configuration: solver'
        )

        # A key that should be in model but is given in run
        override4 = {'run.subset_time': None}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override4, override_groups='simple_supply')

        assert check_error_or_warning(
            excinfo, 'Unrecognised setting in run configuration: subset_time'
        )

        override5 = {
            'run.objective': 'minmax_cost_optimization',
            'run.objective_options': {
                'cost_class': 'monetary',
                'sense': 'minimize',
                'unused_option': 'some_value'
            }
        }

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override5, override_groups='simple_supply')

        assert check_error_or_warning(
            excinfo, 'Objective function argument `unused_option` given but not used by objective function `minmax_cost_optimization`'
        )

    def test_model_version_mismatch(self):
        """
        Model config says model.calliope_version = 0.1, which is not what we
        are running, so we want a warning.
        """
        override = {'model.calliope_version': 0.1}

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, override_groups='simple_supply,one_day')

        assert check_error_or_warning(excinfo, 'Model configuration specifies calliope_version')

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

    def test_tech_as_parent(self):
        """
        All technologies and technology groups must specify a parent
        """

        override1 = AttrDict.from_yaml_string(
            """
            techs.test_supply_tech_parent:
                    essentials:
                        name: Supply tech
                        carrier: gas
                        parent: test_supply_elec
                    constraints:
                        energy_cap_max: 10
                        resource: .inf
            locations.1.test_supply_tech_parent:
            """
        )

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override1, override_groups='simple_supply,one_day')
        check_error_or_warning(error, 'tech `test_supply_tech_parent` has another tech as a parent')

        override2 = AttrDict.from_yaml_string(
            """
            tech_groups.test_supply_group:
                    essentials:
                        carrier: gas
                        parent: test_supply_elec
                    constraints:
                        energy_cap_max: 10
                        resource: .inf
            techs.test_supply_tech_parent.essentials:
                        name: Supply tech
                        parent: test_supply_group
            locations.1.test_supply_tech_parent:
            """
        )

        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=override2, override_groups='simple_supply,one_day')
        check_error_or_warning(error, 'tech_group `test_supply_group` has a tech as a parent')

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
        build_model(override_dict=override_converison_plus('heat'), override_groups='simple_conversion_plus,one_day')


    def test_allowed_time_varying_constraints(self):
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
            with pytest.raises(exceptions.ModelError) as errors:
                build_model(override_dict=override(param), override_groups='simple_storage,one_day')
            assert check_error_or_warning(
                errors,
                'Cannot load `{}` from file for configuration'.format(param)
            )

        # should pass: can have `file=` on the following constraints
        for param in allowed_constraints_file:
            build_model(override_dict=override(param), override_groups='simple_storage,one_day')

    def test_incorrect_location_coordinates(self):
        """
        Either all or no locations must have `coordinates` defined and, if all
        defined, they must be in the same coordinate system (lat/lon or x/y)
        """

        def _override(param0, param1):
            override = {}
            if param0 is not None:
                override.update({'locations.0.coordinates': param0})
            if param1 is not None:
                override.update({'locations.1.coordinates': param1})
            return override

        cartesian0 = {'x': 0, 'y': 1}
        cartesian1 = {'x': 1, 'y': 1}
        geographic0 = {'lat': 0, 'lon': 1}
        geographic1 = {'lat': 1, 'lon': 1}
        fictional0 = {'a': 0, 'b': 1}
        fictional1 = {'a': 1, 'b': 1}

        # should fail: cannot have locations in one place and not in another
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=_override(cartesian0, None), override_groups='simple_storage,one_day')
        check_error_or_warning(error, "Either all or no locations must have `coordinates` defined")

        # should fail: cannot have cartesian coordinates in one place and geographic in another
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=_override(cartesian0, geographic1), override_groups='simple_storage,one_day')
        check_error_or_warning(error, "All locations must use the same coordinate format")

        # should fail: cannot use a non-cartesian or non-geographic coordinate system
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=_override(fictional0, fictional1), override_groups='simple_storage,one_day')
        check_error_or_warning(error, "Unidentified coordinate system")

        # should fail: coordinates must be given as key:value pairs
        with pytest.raises(exceptions.ModelError) as error:
            build_model(override_dict=_override([0, 1], [1, 1]), override_groups='simple_storage,one_day')
        check_error_or_warning(error, "Coordinates must be given in the format")

        # should pass: cartesian coordinates in both places
        build_model(override_dict=_override(cartesian0, cartesian1), override_groups='simple_storage,one_day')

        # should pass: geographic coordinates in both places
        build_model(override_dict=_override(geographic0, geographic1), override_groups='simple_storage,one_day')

    def test_one_way(self):
        """
        With one_way transmission, we remove one direction of a link from
        loc_tech_carriers_prod and the other from loc_tech_carriers_con.
        """
        override = {
            'links.X1,N1.techs.heat_pipes.constraints.one_way': True,
            'links.N1,X2.techs.heat_pipes.constraints.one_way': True,
            'links.N1,X3.techs.heat_pipes.constraints.one_way': True,
            'model.subset_time': '2005-01-01'
        }
        m = calliope.examples.urban_scale(override_dict=override)
        removed_prod_links = ['X1::heat_pipes:N1', 'N1::heat_pipes:X2', 'N1::heat_pipes:X3']
        removed_con_links = ['N1::heat_pipes:X1', 'X2::heat_pipes:N1', 'X3::heat_pipes:N1']

        for link in removed_prod_links:
            assert link not in m._model_data.loc_tech_carriers_prod.values

        for link in removed_con_links:
            assert link not in m._model_data.loc_tech_carriers_con.values

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

    def test_force_resource_ignored(self):
        """
        If a technology is defines force_resource but is not in loc_techs_finite_resource
        it will have no effect
        """

        override = {
            'techs.test_supply_elec.constraints.resource': np.inf,
            'techs.test_supply_elec.constraints.force_resource': True,
        }

        with pytest.warns(exceptions.ModelWarning) as excinfo:
            build_model(override_dict=override, override_groups='simple_supply,one_day')

        assert check_error_or_warning(
            excinfo,
            '`test_supply_elec` at `0` defines force_resource but not a finite resource'
        )

    def test_override_coordinates(self):
        """
        Check that warning is raised if we are completely overhauling the
        coordinate system with an override
        """
        override = {
            'locations': {
                'X1.coordinates': {'lat': 51.4596158, 'lon': -0.1613446},
                'X2.coordinates': {'lat': 51.4652373, 'lon': -0.1141548},
                'X3.coordinates': {'lat': 51.4287016, 'lon': -0.1310635},
                'N1.coordinates': {'lat': 51.4450766, 'lon': -0.1247183}
            },
            'links': {
                'X1,X2.techs.power_lines.distance': 10,
                'X1,X3.techs.power_lines.istance': 5,
                'X1,N1.techs.heat_pipes.distance': 3,
                'N1,X2.techs.heat_pipes.distance': 3,
                'N1,X3.techs.heat_pipes.distance': 4
            }
        }
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            calliope.examples.urban_scale(override_dict=override)

        assert check_error_or_warning(
            excinfo,
            "Updated from coordinate system"
        )


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

    # FIXME: What are the *required* arrays?
    def test_missing_array(self):
        """
        Check that the dataset includes all arrays *required* for a model to function
        """
    # FIXME: What are the *required* attributes?
    def test_missing_attrs(self):
        """
        Check that the dataset includes all attributes *required* for a model to function
        """

    def test_force_infinite_resource(self):
        """
        Ensure that no loc-tech specifies infinite resource and force_resource=True
        """

        override = {
            'techs.test_supply_plus.constraints.resource': 'file=supply_plus_resource_inf.csv',
            'techs.test_supply_plus.constraints.force_resource': True,
        }

        with pytest.raises(exceptions.ModelError) as error_info:
            build_model(override_dict=override, override_groups='simple_supply_plus,one_day')

        assert check_error_or_warning(error_info, 'Ensure all entries are numeric')

    def test_positive_demand(self):
        """
        Resource for demand must be negative
        """

        override = {
            'techs.test_demand_elec.constraints.resource': 'file=demand_elec_positive.csv',
        }

        with pytest.raises(exceptions.ModelError):
            build_model(override_dict=override, override_groups='simple_supply,one_day')

    def test_empty_dimensions(self):
        """
        Empty dimensions lead Pyomo to blow up (building sets with no data),
        so check that we have successfully removed them here.
        """

        model = build_model(override_groups='simple_conversion_plus,one_day')

        assert 'distance' not in model._model_data.data_vars
        assert 'lookup_remotes' not in model._model_data.data_vars

    def check_operate_mode_allowed(self):
        """
        On masking times, operate mode will no longer be allowed
        """

        model = build_model(override_groups='simple_supply,one_day')
        assert model.model_data.attrs['allow_operate_mode'] == 1

        model1 = calliope.examples.time_masking()
        assert model1.model_data.attrs['allow_operate_mode'] == 0

    def test_15min_timesteps(self):

        override = {
            'techs.test_demand_elec.constraints.resource': 'file=demand_elec_15mins.csv',
        }

        model = build_model(override, override_groups='simple_supply,one_day')

        assert model.inputs.timestep_resolution.to_pandas().unique() == [0.25]


class TestUtil():
    def test_concat_iterable_ensures_same_length_iterables(self):
        """
        All iterables must have the same length
        """
        iterables = [('1', '2', '3'), ('4', '5')]
        iterables_swapped = [('4', '5'), ('1', '2', '3')]
        iterables_correct = [('1', '2', '3'), ('4', '5', '6')]
        concatenator = [':', '::']

        with pytest.raises(AssertionError):
            calliope.core.preprocess.util.concat_iterable(iterables, concatenator)
            calliope.core.preprocess.util.concat_iterable(iterables_swapped, concatenator)

        concatenated = calliope.core.preprocess.util.concat_iterable(iterables_correct, concatenator)
        assert concatenated == ['1:2::3', '4:5::6']

    def test_concat_iterable_check_concatenators(self):
        """
        Contatenators should be one shorter than the length of each iterable
        """
        iterables = [('1', '2', '3'), ('4', '5', '6')]
        concat_one = [':']
        concat_two_diff = [':', '::']
        concat_two_same = [':', ':']
        concat_three = [':', ':', ':']

        with pytest.raises(AssertionError):
            calliope.core.preprocess.util.concat_iterable(iterables, concat_one)
            calliope.core.preprocess.util.concat_iterable(iterables, concat_three)

        concatenated1 = calliope.core.preprocess.util.concat_iterable(iterables, concat_two_diff)
        assert concatenated1 == ['1:2::3', '4:5::6']

        concatenated2 = calliope.core.preprocess.util.concat_iterable(iterables, concat_two_same)
        assert concatenated2 == ['1:2:3', '4:5:6']

    def test_vincenty(self):
        # London to Paris: about 344 km
        coords = [(51.507222, -0.1275), (48.8567, 2.3508)]
        distance = calliope.core.preprocess.util.vincenty(coords[0], coords[1])
        assert distance == pytest.approx(343834)  # in meters


class TestTime:
    @pytest.fixture
    def model(self):
        return calliope.examples.urban_scale(
            override_dict={'model.subset_time': ['2005-01-01', '2005-01-10']}
        )

    def test_add_max_demand_timesteps(self, model):
        data = model._model_data_original.copy()
        data = time.add_max_demand_timesteps(data)

        assert (
            data['max_demand_timesteps'].loc[dict(carriers='heat')].values ==
            np.datetime64('2005-01-05T07:00:00')
        )

        assert (
            data['max_demand_timesteps'].loc[dict(carriers='electricity')].values ==
            np.datetime64('2005-01-10T09:00:00')
        )
