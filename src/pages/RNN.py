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
        dcc.Dropdown(id='RNN-dropdown', 
                options=[
            {'label': 'Original Dataset', 'value': 'Original Dataset'},
            {'label': 'Accuracy Plot', 'value': 'Accuracy Plot'},
            {'label': 'Result Plot', 'value': 'Result Plot'},],
            value="Result Plot",), 
    ]), 
    dbc.Row([
        html.Div([
            dcc.Slider(1, 51, 2, 
            value = 51, 
            id = 'resolution_slider', 
            className = "slider")
        ])
    ]),
    dbc.Row([
        dcc.Graph(id='RNN_plot', figure={}) 
    ]), 
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='Running', figure={}) 
        ]), 
        dbc.Col([
            dcc.Graph(id='Running_2', figure={}) 
        ])
    ])
], fluid = True)
