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

    model = calliope.examples.urban_scale()
    model.run()

    plots = ['timeseries', 'capacity', 'transmission']
    basepath = os.path.dirname(__file__)

    for k in plots:
        html = getattr(model.plot, k)(html_only=True)
        path = os.path.join(basepath, 'user', 'images', 'plot_{}.html'.format(k))
        with open(path, 'w') as f:
            f.write(html)
            print('Wrote file: {}'.format(path))
