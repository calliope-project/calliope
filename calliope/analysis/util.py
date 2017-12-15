"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

plotting.py
~~~~~~~~~~~

Functionality to plot model data.

"""

from calliope.core.preprocess.util import vincenty

def get_zoom(coordinate_array, width):
    """
    If mapbox is being used for tranmission plotting, get the zoom based on the
    bounding area of the input data and the width (in pixels) of the map
    """

    # Keys are zoom levels, values are m/pixel at that zoom level
    zoom_dict = {0: 156412, 1: 78206, 2: 39103, 3: 19551, 4: 9776, 5: 4888,
                6: 2444, 7: 1222, 8: 610.984, 9: 305.492, 10: 152.746,
                11: 76.373, 12: 38.187, 13: 19.093, 14: 9.547, 15: 4.773,
                16: 2.387, 17: 1.193, 18: 0.596, 19: 0.298}

    bounds = [coordinate_array.max(dim='locs').values,
              coordinate_array.min(dim='locs').values]

    max_distance = vincenty(*bounds)

    metres_per_pixel = max_distance / width

    for k, v in zoom_dict.items():
        if v > metres_per_pixel:
            continue
        else:
            zoom = k-3
            break

    return zoom
