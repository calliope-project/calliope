import calliope


class TestNationalScaleExampleModel:
    def test_model_initialization_default(self):
        model = calliope.examples.NationalScale()
        model.run()

class TestUrbanScaleExampleModel:
    def test_model_initialization_default(self):
        model = calliope.examples.UrbanScale()
        model.run()
