"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.backend.pyomo.model import generate_model


def run(model_data):
    backend_model = generate_model(model_data)
    pass

def get_results(backend_model):
    pass
