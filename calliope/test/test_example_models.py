from pytest import approx

import calliope


class TestNationalScaleExampleModel:
    def test_preprocess_national_scale(self):
        model = calliope.examples.national_scale()

    def test_preprocess_time_clustering(self):
        model = calliope.examples.time_clustering()

    def test_preprocess_time_resampling(self):
        model = calliope.examples.time_resampling()


class TestUrbanScaleExampleModel:
    def test_preprocess_urban_scale(self):
        model = calliope.examples.urban_scale()

    def test_preprocess_milp(self):
        model = calliope.examples.milp()


class TestNationalScaleExampleModelSenseChecks:
    def test_nationalscale_example_results(self):
        model = calliope.examples.national_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()

        assert model.results.storage_cap.to_pandas()['region1-1::csp'] == approx(45129.950)
        assert model.results.storage_cap.to_pandas()['region2::battery'] == approx(6675.173)

        assert model.results.energy_cap.to_pandas()['region1-1::csp'] == approx(4.626588e+03)
        assert model.results.energy_cap.to_pandas()['region2::battery'] == approx(1000)
        assert model.results.energy_cap.to_pandas()['region1::ccgt'] == approx(30000)

        #assert float(model.results.cost.sum()) == approx(38997.3544)


class TestUrbanScaleExampleModelSenseChecks:
    def test_urbanscale_example_results(self):
        model = calliope.examples.urban_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()
