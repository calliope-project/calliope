import os
import calliope
import pytest  # pylint: disable=unused-import

from calliope.core.attrdict import AttrDict
import calliope.exceptions as exceptions


HTML_STRINGS = AttrDict.from_yaml(
    os.path.join(os.path.dirname(__file__), 'common', 'html_strings.yaml')
)


class TestPlotting:

    def test_national_scale_plotting(self):
        override = {'model.subset_time': '2005-01-01'}
        model = calliope.examples.national_scale(override_dict=override)
        model.run()

        plot_html_outputs = {
            'capacity': model.plot.capacity(html_only=True),
            'timeseries': model.plot.timeseries(html_only=True),
            'transmission': model.plot.transmission(html_only=True),
        }

        for plot_type in HTML_STRINGS['national_scale']:
            for string in HTML_STRINGS['national_scale'][plot_type]:
                assert string in plot_html_outputs[plot_type]

        # Also just try plotting the summary
        model.plot.summary()

    def test_milp_plotting(self):
        override = {'model.subset_time': '2005-01-01'}
        model = calliope.examples.milp(override_dict=override)
        model.run()

        plot_html_outputs = {
            'capacity': model.plot.capacity(html_only=True),
            'timeseries': model.plot.timeseries(html_only=True),
            'transmission': model.plot.transmission(html_only=True),
        }

        for plot_type in HTML_STRINGS['milp']:
            for string in HTML_STRINGS['milp'][plot_type]:
                assert string in plot_html_outputs[plot_type]

        # Also just try plotting the summary
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
