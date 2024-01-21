# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""
Generate interactive plots to use within the docs
"""

import tempfile
from pathlib import Path

import calliope
import pandas as pd
import plotly.graph_objects as go
from mkdocs.structure.files import File

TEMPDIR = tempfile.TemporaryDirectory()


def on_files(files: list, config: dict, **kwargs):
    """Generate schema markdown reference sheets and attach them to the documentation."""

    file_obj = _generate_front_page_timeseries_plot(config)
    files.append(file_obj)

    return files


def _generate_front_page_timeseries_plot(config: dict) -> File:
    """Generate a timeseries plot from urban-scale example model outputs with dropdown menu per model carrier.

    Args:
        config (dict): mkdocs config dict.

    Returns:
        File: File object to add to mkdocs file list.
    """
    model = calliope.examples.urban_scale()
    model.build()
    model.solve()

    carriers = model.inputs.carriers.values

    colors = model.inputs.color.groupby(model.inputs.name).first("techs").to_series()
    df_demand = _get_net_flows(model, techs=model.inputs.parent == "demand")
    df_flows_other = _get_net_flows(model, techs=model.inputs.parent != "demand")

    fig = go.Figure()
    visible = True

    # Buttons for the dropdown menu
    buttons = [
        {
            "label": carrier,
            "method": "update",
            "args": [{"visible": []}, {"title": f"{carrier.title()} flows"}],
        }
        for carrier in carriers
    ]

    # Can't use plotly express if we want to build a figure with a dropdown.
    for carrier in carriers:
        n_techs = 0
        for tech in df_flows_other.name.unique():
            _df = df_flows_other[
                (df_flows_other.carriers == carrier) & (df_flows_other.name == tech)
            ]
            _color = colors[tech]
            fig.add_trace(
                go.Bar(
                    x=_df["timesteps"],
                    y=_df["flow"],
                    marker_color=_color,
                    name=tech,
                    legendgroup=tech,
                    visible=visible,
                )
            )
            n_techs += 1

        _df = df_demand[df_demand.carriers == carrier]
        fig.add_trace(
            go.Scatter(
                x=_df["timesteps"],
                y=-1 * _df["flow"],
                marker_color="black",
                name="Demand",
                legendgroup="demand",
                visible=visible,
            )
        )
        visible = False

        for button in buttons:
            if button["label"] == carrier:
                button["args"][0]["visible"].extend([True for i in range(n_techs + 1)])
            else:
                button["args"][0]["visible"].extend([False for i in range(n_techs + 1)])

    fig.update_layout(
        barmode="relative",
        yaxis={"title": "Flow in/out (kWh)"},
        title={"text": buttons[0]["args"][1]["title"], "xanchor": "center", "x": 0.5},
        updatemenus=[
            {
                "active": 0,
                "buttons": buttons,
                "xanchor": "left",
                "x": 0,
                "y": 1,
                "yanchor": "bottom",
            }
        ],
    )
    output_path = Path(TEMPDIR.name) / "img" / "plotly_frontpage_timeseries.html"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(
        fig.to_html(include_plotlyjs="cdn", full_html=False, include_mathjax=False)
    )

    return File(
        path=str(output_path.relative_to(TEMPDIR.name)),
        src_dir=TEMPDIR.name,
        dest_dir=config["site_dir"],
        use_directory_urls=config["use_directory_urls"],
    )


def _get_net_flows(model: calliope.Model, **sel) -> pd.DataFrame:
    """Generate a tidy dataframe of net flows (flows out - flows in).

    All flows are summed over nodes.
    Technologies are renamed according to their "long" name and summed over any matching names (e.g. transmission).
    All net-zero flows are NaNed and dropped from the tidy dataframe.

    Args:
        model (calliope.Model): Calliope model with results.
    Keyword Args:
        Index items on which to slice the flows in xarray before summing over nodes and renaming techs.
    Returns:
        pd.DataFrame: Net-flow timeseries tidy dataframe.
    """

    if "techs" in sel.keys():
        names = model.inputs.name.sel(techs=sel["techs"])
    else:
        names = model.inputs.name
    return (
        (model.results.flow_out.fillna(0) - model.results.flow_in.fillna(0))
        .sel(**sel)
        .sum("nodes", min_count=1)
        .groupby(names)
        .sum("techs")
        .to_series()
        .where(lambda x: x != 0)
        .dropna()
        .to_frame("flow")
        .reset_index()
    )
