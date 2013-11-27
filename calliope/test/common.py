"""Common functions used in tests"""


import cStringIO as StringIO

import calliope


def assert_almost_equal(x, y, tolerance=0.0001):
    assert abs(x-y) < tolerance


def simple_model(config_techs='test/common/techs_minimal.yaml',
                 config_nodes='test/common/nodes_minimal.yaml'):
    config_run = """
    input:
        techs: {techs}
        nodes: {nodes}
        path: 'test/common'
    output:
        save: false
    """
    # Fill in `techs` and `nodes`
    config_run = config_run.format(techs=config_techs, nodes=config_nodes)
    config_run = StringIO.StringIO(config_run)  # Make it a file object
    return calliope.Model(config_run)
