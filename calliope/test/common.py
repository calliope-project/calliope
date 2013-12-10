"""Common functions used in tests"""


import os

import calliope


solver = 'cplex'  # TODO this needs to be done differently


def assert_almost_equal(x, y, tolerance=0.0001):
    assert abs(x-y) < tolerance


def _add_test_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def simple_model(config_model=None, config_techs=None, config_nodes=None,
                 path=None, config_run=None, override=None):
    if not config_model:
        config_model = _add_test_path('common/model_minimal.yaml')
    if not config_techs:
        config_techs = _add_test_path('common/techs_minimal.yaml')
    if not config_nodes:
        config_nodes = _add_test_path('common/nodes_minimal.yaml')
    if not path:
        path = _add_test_path('common/t_1h')
    if not config_run:
        config_run = """
        mode: plan
        input:
            model: '{model}'
            techs: '{techs}'
            nodes: '{nodes}'
            path: '{path}'
        output:
            save: false
        """
    # Fill in `techs` and `nodes`
    config_run = config_run.format(model=config_model, techs=config_techs,
                                   nodes=config_nodes, path=path)
    # Make it an AttrDict
    config_run = calliope.utils.AttrDict.from_yaml_string(config_run)
    return calliope.Model(config_run, override)
