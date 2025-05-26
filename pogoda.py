import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import datetime as dt

# Jeśli tools z plotly nie jest używany, można go usunąć
# from plotly import tools

df = pd.read_csv(
    'dataexport_20200613T163949.csv',
    skiprows=9,
    index_col=0,
    parse_dates=True
)
