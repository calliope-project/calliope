import pytest

from calliope.test.common.util import build_test_model as build_model

@pytest.fixture(scope="session")
def simple_supply():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_milp():
    m = build_model({}, "supply_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def storage_milp():
    m = build_model({}, "storage_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def conversion_plus_milp():
    m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_and_supply_plus_milp():
    m = build_model({}, "supply_and_supply_plus_milp,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_supply_and_supply_plus():
    m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_storage():
    m = build_model({}, "simple_storage,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_conversion():
    m = build_model({}, "simple_conversion,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_export():
    m = build_model({}, "supply_export,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def supply_purchase():
    m = build_model({}, "supply_purchase,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def conversion_plus_purchase():
    m = build_model({}, "conversion_plus_purchase,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def storage_purchase():
    m = build_model({}, "storage_purchase,two_hours,investment_costs")
    m.run()
    return m


@pytest.fixture(scope="session")
def simple_conversion_plus():
    m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
    m.run()
    return m