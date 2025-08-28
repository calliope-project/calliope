"""Util packaging."""

import numpy as np

DTYPE_OPTIONS = {
    "string": str,
    "float": float,
    "bool": bool,
    "datetime": np.datetime64,
    "date": np.datetime64,
    "integer": int,
}

DATETIME_DTYPE = "M"
"""Numpy type kind for datetime arrays"""
