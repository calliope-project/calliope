# from calliope.core.attrdict import AttrDict
# from calliope.test import common
# from calliope.test.common import solver, solver_io, _add_test_path


# def create_and_run_model(override, iterative_warmstart=False):
#     locations = """
#         locations:
#             1:
#                 techs:
#                     ccgt:
#                         constraints:
#                             energy_cap_max: 100
#                     demand_power:
#                         constraints:
#                             resource: file=demand-blocky_r.csv:demand
#         links:
#     """
#     config_run = """
#         mode: plan
#         model: ['{techs}', '{locations}']
#     """
#     override = AttrDict.from_yaml_string(override)
#     override.set_key('solver', solver)
#     override.set_key('solver_io', solver_io)
#     with tempfile.NamedTemporaryFile(delete=False) as f:
#         f.write(locations.encode('utf-8'))
#         f.read()
#         model = common.simple_model(config_run=config_run,
#                                     config_locations=f.name,
#                                     override=override,
#                                     path=_add_test_path('common/t_time'))
#     model.run(iterative_warmstart)
#     return model

class TestModelRun:
    def override_base_technology_groups(self):
        """
        Test that base technology are not being overriden by any user input
        """

    def undefined_carriers(self):
        """
        Test that user has input either carrier or carrier_in/_out for each tech
        """

    def incorrect_time_subset(self):
        """
        If subset_time is a list, it must have two entries (start_time, end_time)
        If subset_time is not a list, it should successfully subset on the given
        string/integer
        """

    def incorrect_date_format(self):
        """
        Test the date parser catches a different date format from file than
        user input/default (inc. if it is just one line of a file that is incorrect)
        """

    def inconsistent_time_indeces(self):
        """
        Test that, including after any time subsetting, the indeces of all time
        varying input data are consistent with each other
        """

    def empty_key_on_explode(self):
        """
        On exploding locations (from ``'1--3'`` or ``'1,2,3'`` to
        ``['1', '2', '3']``), raise error on the resulting list being empty
        """

    def key_clash_on_set_loc_key(self):
        """
        Raise error on attempted overwrite of information regarding a recently
        exploded location
        """


class TestChecks:
    def unknown_carrier_tier(self):
        """
        User can only use 'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3',
        'out_3', 'ratios']
        """

    def name_overlap(self):
        """
        No tech_groups/techs may have the same identifier as the built-in groups
        """

    def unspecified_parent(self):
        """
        All technologies and technology groups must specify a parent
        """

    def resource_as_carrier(self):
        """
        No carrier in technology or technology group can be called `resource`
        """

    def missing_required_constraints(self):
        """
        A technology within an abstract base technology must define a subset of
        hardcoded constraints in order to function
        """

    def defining_non_allowed_constraints(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded constraints, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """

    def defining_non_allowed_costs(self):
        """
        A technology within an abstract base technology can only define a subset
        of hardcoded costs, anything else will not be implemented, so are
        not allowed for that technology. This includes misspellings
        """

    def exporting_unspecified_carrier(self):
        """
        User can only define an export carrier if it is defined in
        ['carrier_out', 'carrier_out_2', 'carrier_out_3']
        """

    def allowed_time_varying_constraints(self):
        """
        `file=` is only allowed on a hardcoded list of constraints, unless
        `_time_varying` is appended to the constraint (i.e. user input)
        """

    def incorrect_location_coordinates(self):
        """
        Either all or no locations must have `coordinates` defined and, if all
        defined, they must be in the same coordinate system (lat/lon or x/y)
        """


class TestDataset:
    def inconsistent_timesteps(self):
        """
        Timesteps must be consistent?
        """


class TestUtil():
    def concat_iterable_ensures_same_length_iterables(self):
        """
        All iterables must have the same length
        """

    def concat_iterable_check_concatenators(self):
        """
        Contatenators be one shorter than the length of each iterable
        """
