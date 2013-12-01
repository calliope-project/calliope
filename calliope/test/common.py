"""Common functions used in tests"""


import cStringIO as StringIO
import os

import calliope


def assert_almost_equal(x, y, tolerance=0.0001):
    assert abs(x-y) < tolerance


def _add_test_path(path):
    return os.path.join(os.path.dirname(__file__), path)


def simple_model(config_techs=None, config_nodes=None, path=None,
                 config_run=None):
    if not config_techs:
        config_techs = _add_test_path('common/techs_minimal.yaml')
    if not config_nodes:
        config_nodes = _add_test_path('common/nodes_minimal.yaml')
    if not path:
        path = _add_test_path('common/t_1h')
    if not config_run:
        config_run = """
        input:
            techs: '{techs}'
            nodes: '{nodes}'
            path: '{path}'
        output:
            save: false
        """
    # Fill in `techs` and `nodes`
    config_run = config_run.format(techs=config_techs, nodes=config_nodes,
                                   path=path)
    config_run = StringIO.StringIO(config_run)  # Make it a file object
    return calliope.Model(config_run)
