"""Development file for running Calliope model with time seres."""

import pandas as pd
import calliope
import pdb


mycsp = pd.read_csv('mycsp.csv')
model = calliope.Model('../example_models/AH/model.yaml')
