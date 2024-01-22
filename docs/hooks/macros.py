import pandas as pd


def define_env(env):
    "Hook function"

    @env.macro
    def read_csv(file, **kwargs):
        """
        Read a CSV file and render it as a HTML table.
        """
        styles = [
            # table properties
            dict(
                selector=" ",
                props=[
                    ("margin", "0"),
                    ("font-size", "80%"),
                    ("font-family", '"Helvetica", "Arial", sans-serif'),
                    ("border-collapse", "collapse"),
                    ("border", "none"),
                ],
            ),
            # background shading
            dict(
                selector="tbody tr:nth-child(even)",
                props=[("background-color", "#fff")],
            ),
            dict(
                selector="tbody tr:nth-child(odd)", props=[("background-color", "#eee")]
            ),
            # cell spacing
            dict(selector="td", props=[("text-align", "left"), ("padding", "0.2em")]),
            # header cell properties
            dict(
                selector="th",
                props=[
                    ("vertical-align", "middle"),
                    ("padding", "0.2em"),
                    ("white-space", "nowrap"),
                ],
            ),
            dict(selector="th.row_heading", props=[("text-align", "right")]),
            dict(selector="th.index_name", props=[("text-align", "right")]),
            dict(selector=".col_heading", props=[("text-align", "left")]),
        ]
        return (
            pd.read_csv(file, **kwargs)
            .head(2)
            .style.set_table_styles(styles)
            .to_html(sparse_columns=False, sparse_index=False)
        )
