import os
import calliope
import pytest  # pylint: disable=unused-import

from calliope.core.attrdict import AttrDict
import calliope.exceptions as exceptions
import pandas as pd

this_path = os.path.dirname(__file__)

html_strings_location = os.path.join(this_path, 'common', 'html_strings.yaml')

html_strings = AttrDict.from_yaml(html_strings_location)


class TestPlotting:

    def test_national_scale_plotting(self):
        override = {'model.subset_time': '2005-01-01'}
        model = calliope.examples.national_scale(override_dict=override)
        model.run()

        capacity_string = model.plot.capacity(html_only=True)
        timeseries_string = model.plot.timeseries(html_only=True)
        transmission_string = model.plot.transmission(html_only=True)

        # Div IDs are always unique, so we ignore the string until the start of the data
        assert capacity_string[capacity_string.find('[{'):] == html_strings['national_scale']['capacity']
        assert timeseries_string[timeseries_string.find('[{'):] == html_strings['national_scale']['timeseries']
        assert transmission_string[transmission_string.find('[{'):] == html_strings['national_scale']['transmission']

    def test_milp_plotting(self):
        override = {'model.subset_time': '2005-01-01'}
        model = calliope.examples.milp(override_dict=override)
        model.run()

        capacity_string = model.plot.capacity(html_only=True)
        timeseries_string = model.plot.timeseries(html_only=True)
        transmission_string = model.plot.transmission(html_only=True)

        # Div IDs are always unique, so we ignore the string until the start of the data
        assert capacity_string[capacity_string.find('[{'):] == html_strings['milp']['capacity']
        assert timeseries_string[timeseries_string.find('[{'):] == html_strings['milp']['timeseries']
        assert transmission_string[transmission_string.find('[{'):] == html_strings['milp']['transmission']

        # Just try plotting the summary
        model.plot.summary()

    def test_failed_cap_plotting(self):
        override = {'model.subset_time': '2005-01-01'}
        model = calliope.examples.national_scale(override_dict=override)
        model.run()

        # should fail, not in array
        with pytest.raises(exceptions.ModelError):
            model.plot.capacity(array='carrier_prod')
            model.plot.capacity(array=['energy_eff', 'energy_cap'])
            # orient has to be 'v', 'vertical', 'h', or 'horizontal'
            model.plot.capacity(orient='g')

    def test_failed_timeseries_plotting(self):
        override = {'model.subset_time': '2005-01-01'}
        model = calliope.examples.national_scale(override_dict=override)
        model.run()

        # should fail, not in array
        with pytest.raises(exceptions.ModelError):
            model.plot.timeseries(array='energy_cap')
            model.plot.timeseries(squeeze=False)
            model.plot.timeseries(sum_dims=None)