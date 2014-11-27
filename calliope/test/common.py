"""Common functions used in tests"""

import os

import calliope

# README: Hardcoding GLPK here which needs to be installed for tests to pass
# Should also work with other solvers, e.g. 'cplex' or 'gurobi'
solver = 'glpk'  # this gets imported by individual test_model_* files


def assert_almost_equal(x, y, tolerance=0.0001):
    assert abs(x - y) < tolerance


def _add_test_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def simple_model(config_model=None, config_techs=None, config_locations=None,
                 path=None, config_run=None, override=None):
    if not config_model:
        config_model = _add_test_path('common/model_minimal.yaml')
    if not config_techs:
        config_techs = _add_test_path('common/techs_minimal.yaml')
    if not config_locations:
        config_locations = _add_test_path('common/locations_minimal.yaml')
    if not path:
        path = _add_test_path('common/t_1h')
    if not override:
        override = calliope.utils.AttrDict({'override.data_path': path})
    else:
        override.set_key('override.data_path', path)
    if not config_run:
        config_run = """
        model: ['{model}', '{techs}', '{locations}']
        mode: plan
        """
    # Fill in `techs` and `locations`
    config_run = config_run.format(model=config_model, techs=config_techs,
                                   locations=config_locations)
    # Make it an AttrDict
    config_run = calliope.utils.AttrDict.from_yaml_string(config_run)
    return calliope.Model(config_run, override)
