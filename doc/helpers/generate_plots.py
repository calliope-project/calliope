"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_plots.py
~~~~~~~~~~~~~~~~~

Generate plotly plots to include in the documentation.

"""

import os

import calliope


_TOKEN = 'pk.eyJ1IjoiY2FsbGlvcGUtcHJvamVjdCIsImEiOiJjamVwd280ODkwYzh6Mnhxbm1qYnU4bWI4In0.mv2O1aDqQEkOUwAIcVoUMA'


def model_plots(
        model,
        plots=['timeseries', 'capacity', 'transmission', 'transmission_token'],
        filename_prefix=None):

    basepath = os.path.dirname(__file__)

    for k in plots:
        if k == 'transmission_token':
            html = getattr(model.plot, 'transmission')(html_only=True, mapbox_access_token=_TOKEN)
        else:
            html = getattr(model.plot, k)(html_only=True)

        filename = 'plot_{}.html'.format(k)

        if filename_prefix:
            filename = str(filename_prefix) + filename

        path = os.path.join(basepath, '..', 'user', 'images', filename)
        with open(path, 'w') as f:
            f.write(html)
            print('Wrote file: {}'.format(path))


def generate_all_plots():
    override_path = os.path.join(os.path.dirname(calliope.__file__), 'example_models', 'urban_scale')
    model_urban = calliope.examples.urban_scale(
        override_file=os.path.join(override_path, 'overrides.yaml:mapbox_ready')
    )
    model_urban.run()
    model_plots(model_urban)

    model_clustered = calliope.examples.time_clustering()
    model_clustered.run()
    model_plots(model_clustered, plots=['timeseries'], filename_prefix='clustered_')


if __name__ == '__main__':
    generate_all_plots()
