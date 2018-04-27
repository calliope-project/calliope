import os
import calliope
import pytest  # pylint: disable=unused-import
import tempfile

from calliope.core.attrdict import AttrDict
from calliope.test.common.util import check_error_or_warning


HTML_STRINGS = AttrDict.from_yaml(
    os.path.join(os.path.dirname(__file__), 'common', 'html_strings.yaml')
)


class TestPlotting:
    @pytest.fixture(scope="module")
    def national_scale_example(self):
        model = calliope.examples.national_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()
        return model

    def test_national_scale_plotting(self, national_scale_example):
        model = national_scale_example

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

        # Testing that the model can handle not having supply_plus technologies
        # Wrapped in temporary directory as we can't stop it saving an HTML file
        model._model_data = model._model_data.drop('resource_con')
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'test_plot.html')
            model.plot.timeseries(
                plotly_kwarg_updates={'auto_open': False, 'filename': out_path}
            )

    def test_milp_plotting(self):
        override = {'model.subset_time': '2005-07-01'}
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

    def test_failed_cap_plotting(self, national_scale_example):
        model = national_scale_example

        # should fail, not in array
        with pytest.raises(ValueError):
            model.plot.capacity(array='carrier_prod')
            model.plot.capacity(array=['energy_eff', 'energy_cap'])
            # orient has to be 'v', 'vertical', 'h', or 'horizontal'
            model.plot.capacity(orient='g')

    def test_failed_timeseries_plotting(self, national_scale_example):
        model = national_scale_example

        # should fail, not in array
        with pytest.raises(ValueError):
            model.plot.timeseries(array='energy_cap')
            model.plot.timeseries(squeeze=False)
            model.plot.timeseries(sum_dims=None)

    def test_clustered_plotting(self):
        override = {'model.time.function_options.k': 2}
        model = calliope.examples.time_clustering(override_dict=override)

        plot_html = model.plot.timeseries(html_only=True)
        for string in HTML_STRINGS['clustering']['timeseries']:
            assert string in plot_html

        # While we have a model that hasn't been run, try plotting transmission and capacity
        model.plot.transmission(html_only=True)
        model.plot.capacity(html_only=True)

    def test_subset_plotting(self, national_scale_example):
        model = national_scale_example

        model.plot.capacity(
            html_only=True, subset={'timesteps': ['2015-01-01 01:00']}
        )

        # should raise, subsetting with a tech that does not exist
        with pytest.raises(ValueError) as excinfo:
            model.plot.capacity(
                html_only=True, subset={'techs': ['foobar']}
            )

        assert check_error_or_warning(excinfo, 'No data to plot')

    def test_subset_array(self, national_scale_example):
        model = national_scale_example

        model.plot.capacity(html_only=True, array='inputs')
        model.plot.capacity(html_only=True, array='results')
        model.plot.capacity(html_only=True, array='energy_cap')
        model.plot.capacity(html_only=True, array='storage_cap')
        model.plot.capacity(
            html_only=True, array=['systemwide_levelised_cost', 'storage_cap']
        )

        model.plot.timeseries(html_only=True, array='inputs')
        model.plot.timeseries(html_only=True, array='results')
        model.plot.timeseries(html_only=True, array='power')
        model.plot.timeseries(html_only=True, array='resource')
        model.plot.timeseries(
            html_only=True, array=['resource_con', 'cost_var']
        )

    def test_long_name(self, national_scale_example):
            model = national_scale_example
            model._model_data['names'] = model._model_data.names.astype('<U100')
            model._model_data.names.loc['ccgt'] = (
                'a long name for a technology, longer than 30 characters'
            )
            model._model_data.names.loc['csp'] = (
                'a really very long name for a technology that is longer than 60 characters'
            )
            model._model_data.names.loc['battery'] = (
                'another_long_name_but_without_space_to_break_at'
            )
            model._model_data.names.loc['ac_transmission'] = (
                'long_transmission_name_which_has two break types in technology name'
            )

            broken_names = [
                'a long name for a technology,<br>longer than 30 characters',
                'another_long_name_but_without_...<br>space_to_break_at',
                'a really very long name for a<br>technology that is longer<br>than 60 characters'
            ]

            html_cap = model.plot.capacity(html_only=True)
            html_timeseries = model.plot.timeseries(html_only=True)
            html_transmission = model.plot.transmission(html_only=True)
            for i in broken_names:
                assert i in html_cap
                assert i in html_timeseries
            assert (
                'long_transmission_name_which_h...<br>as two break types in<br>technology name'
                in html_transmission
            )

    def test_plot_cost(self):
        model = calliope.examples.national_scale(
            override_dict={
                'techs.ccgt.costs.carbon': {'energy_cap': 10, 'interest_rate': 0.01}
            }
        )

        model.run()

        # should fail, multiple costs provided, can only plot one
        with pytest.raises(ValueError):
            model.plot.capacity(html_only=True, array='results')

        # should succeed, multiple costs provided, subset to one
        model.plot.capacity(
            html_only=True, array='results', subset={'costs': 'carbon'}
        )

        # FIXME: sum_dims doesn't seem to work at all
        # model.plot.capacity(html_only=True, sum_dims=['locs'])

    def test_to_file(self, national_scale_example):
        model = national_scale_example

        # should fail, 'gif' not in allowed extensions
        with pytest.raises(TypeError):
            model.plot.capacity(
                to_file='plot_to_save.gif',
                plotly_kwarg_updates={'auto_open': False})

        # FIXME: currently throws up save dialogue rather than just
        # saving the file

        #with tempfile.TemporaryDirectory() as tempdir:
        #    for extension in ['png', 'jpeg', 'svg', 'webp']:
        #        out_path = os.path.join(tempdir, 'plot_to_save.' + extension)
        #        model.plot.capacity(array='energy_cap', to_file=out_path, auto_open=False)
        #        assert os.path.isfile(out_path)

        # should fail, cannot save a plot with multiple DataArrays being plotted
        with pytest.raises(ValueError):
            model.plot.capacity(
                to_file='plot_to_save.svg',
                plotly_kwarg_updates={'auto_open': False})

        # test saving summary to file
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'test_summary.html')
            model.plot.summary(to_file=out_path)
            assert os.path.isfile(out_path)
