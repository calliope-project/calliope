"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_tutorials.py
~~~~~~~~~~~~~~~~~

Generate HTML for Jupyter notebook tutorials.

"""

if __name__ == '__main__':
    import os
    import nbconvert
    import nbformat

    # TODO: remove padding from #notebook-container
    # TODO: set width of #notebook-container.container to 100%
    # TODO: set div#notebook font-size to 12px?
    # TODO: remove .prompt padding-left and padding-right & set wdith to 10ex
    # TODO: remove 'border' from table and set border properties for rendered_html th, tr, and td

    basepath = os.path.dirname(__file__)
    tutorial_path = os.path.join(basepath, '..', '_static', 'notebooks')
    tutorials = [i for i in os.listdir(tutorial_path) if i.endswith('.ipynb')]


    for k in tutorials:
        notebook_path = os.path.join(tutorial_path, k)
        html_path = os.path.join(tutorial_path, k.replace('.ipynb', '.html'))

        nb = nbformat.read(notebook_path, 4)
        html, resource = nbconvert.exporters.export(nbconvert.exporters.get_exporter('html'), nb)

        # Remove plotly javascript
        start = html.find(
            """<div class="output_html rendered_html output_subarea ">\n<script type='text/javascript'>if(!window.Plotly)"""
        )
        end = html.find(
            """});require(['plotly'], function(Plotly) {window.Plotly = Plotly;});}</script>\n</div>"""
        )

        html_new = html[:start] + html[end + 85:]

        # remove call to internal javascript from plotly plots
        html_new = html_new.replace('require(["plotly"], function(Plotly) {', '')
        html_new = html_new.replace(')});</script>', ');</script>')

        # Write to file
        with open(html_path, 'w', encoding="utf-8") as file:
            file.write(html_new)