import pytest  # pylint: disable=unused-import

import calliope


class TestTime:
    @pytest.fixture
    def model(self):
        return calliope.examples.national_scale()

    def test_add_max_demand_timesteps(self, model):
        pass
