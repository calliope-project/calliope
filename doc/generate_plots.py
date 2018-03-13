"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_plots.py
~~~~~~~~~~~~~~~~~

Generate plotly plots to include in the documentation.

"""

if __name__ == '__main__':
    import os

    import calliope
    override_path = os.path.join(os.path.dirname(calliope.__file__), 'example_models', 'urban_scale')
    model = calliope.examples.urban_scale(
        override_file=os.path.join(override_path, 'overrides.yaml:mapbox_ready')
    )

    model.run()

    plots = ['timeseries', 'capacity', 'transmission', 'transmission_token']
    basepath = os.path.dirname(__file__)
    token = 'pk.eyJ1IjoiY2FsbGlvcGUtcHJvamVjdCIsImEiOiJjamVwd280ODkwYzh6Mnhxbm1qYnU4bWI4In0.mv2O1aDqQEkOUwAIcVoUMA'

    for k in plots:
        if k == 'transmission_token':
            html = getattr(model.plot, 'transmission')(html_only=True, mapbox_access_token=token)
        else:
            html = getattr(model.plot, k)(html_only=True)
        path = os.path.join(basepath, 'user', 'images', 'plot_{}.html'.format(k))
        with open(path, 'w') as f:
            f.write(html)
            print('Wrote file: {}'.format(path))
