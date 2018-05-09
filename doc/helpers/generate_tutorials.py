"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_tutorials.py
~~~~~~~~~~~~~~~~~

Generate HTML for Jupyter notebook tutorials.

"""

import os
import nbconvert
import nbformat


def generate_tutorials():
    basepath = os.path.dirname(__file__)
    tutorial_path = os.path.join(basepath, '..', '_static', 'notebooks')
    tutorials = [i for i in os.listdir(tutorial_path) if i.endswith('.ipynb')]

    for k in tutorials:
        notebook_path = os.path.join(tutorial_path, k)
        html_path = os.path.join(tutorial_path, k.replace('.ipynb', '.html'))

        nb = nbformat.read(notebook_path, 4)
        html, resource = nbconvert.exporters.export(nbconvert.exporters.get_exporter('html'), nb)

        # Remove plotly javascript
        start_string = """<div class="output_html rendered_html output_subarea ">\n<script type='text/javascript'>if(!window.Plotly)"""
        start = html.find(start_string)
        end_string = """});require(['plotly'], function(Plotly) {window.Plotly = Plotly;});}</script>\n</div>"""
        end = html.find(end_string)

        html_new = html[:start] + html[end + len(end_string):]

        # remove call to internal javascript from plotly plots
        html_new = html_new.replace('require(["plotly"], function(Plotly) {', '')
        html_new = html_new.replace(')});</script>', ');</script>')

        # Also get rid of table borders
        html_new = html_new.replace('border="1" ', '')

        # Remove all inline CSS styles
        # NOTE: this only works if <style> and </style> are on lines of their own
        html_lines = html_new.split('\n')
        start_lines = []
        end_lines = []
        for i, line in enumerate(html_lines):
            if '<style' in line:
                start_lines.append(i)
            if '</style>' in line:
                end_lines.append(i)

        assert len(start_lines) == len(end_lines)

        for i, line in enumerate(start_lines):
            del html_lines[line:end_lines[i]]

        html_new = '\n'.join(html_lines)

        # Write to file
        with open(html_path, 'w', encoding="utf-8") as file:
            file.write(html_new)


if __name__ == '__main__':
    generate_tutorials()
