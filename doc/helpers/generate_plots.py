# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
generate_plots.py
~~~~~~~~~~~~~~~~~

Generate plotly plots to include in the documentation.

"""

import os

import calliope

_TOKEN = "pk.eyJ1IjoiY2FsbGlvcGUtcHJvamVjdCIsImEiOiJjamVwd280ODkwYzh6Mnhxbm1qYnU4bWI4In0.mv2O1aDqQEkOUwAIcVoUMA"


def model_plots(
    model,
    plots=["timeseries", "capacity", "transmission", "transmission_token", "flows"],
    filename_prefix=None,
    out_path=None,
):
    basepath = os.path.dirname(__file__)

    for k in plots:
        if k == "transmission_token":
            html = getattr(model.plot, "transmission")(
                html_only=True, mapbox_access_token=_TOKEN
            )
        elif k == "summary":
            html = getattr(model.plot, k)()
        else:
            html = getattr(model.plot, k)(html_only=True)

        filename = "plot_{}.html".format(k)

        if filename_prefix:
            filename = str(filename_prefix) + filename

        if out_path is None:
            out_path = os.path.join(basepath, "..", "user", "images")

        path = os.path.join(out_path, filename)
        with open(path, "w") as f:
            f.write(html)
            print("Wrote file: {}".format(path))


def generate_all_plots():
    model_urban = calliope.examples.urban_scale(scenario="mapbox_ready")
    model_urban.build()
    model_urban.solve()
    model_plots(model_urban)
    model_plots(
        model_urban,
        plots=["summary"],
        out_path=os.path.join(os.path.dirname(__file__), "..", "_static"),
    )

    model_clustered = calliope.examples.time_clustering()
    model_clustered.build()
    model_clustered.solve()
    model_plots(model_clustered, plots=["timeseries"], filename_prefix="clustered_")


if __name__ == "__main__":
    generate_all_plots()
