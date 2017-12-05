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