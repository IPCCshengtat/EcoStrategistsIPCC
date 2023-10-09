from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash as dash
from dash import dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pymoo.mcdm.pseudo_weights import PseudoWeights
import time

dash.register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dcc.Dropdown(id='ANN-dropdown', 
                options=[
            {'label': 'Population Dataset', 'value': 'Population Dataset'},
            {'label': 'GDP Dataset', 'value': 'GDP Dataset'},
            {'label': 'Accuracy Plot', 'value': 'Accuracy Plot'},
            {'label': 'Result Plot', 'value': 'Result Plot'},],
            value="Result Plot",), 
    ]), 
    dbc.Row([
        html.Div([
            dcc.Slider(1, 51, 2, 
            value = 51, 
            id = 'resolution_slider_ANN', 
            className = "slider")
        ])
    ]),
    dbc.Row([
        dcc.Graph(id='ANN_plot', figure={}) 
    ]), 
], fluid = True)
