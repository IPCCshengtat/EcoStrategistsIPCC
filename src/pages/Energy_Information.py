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
        dcc.Dropdown(id='CO2_map', 
                options=[
            {'label': 'Electricity Consumption in Malaysia', 'value': 'Electricity Consumption in Malaysia'},
            {'label': 'Renewable energy share in the total final energy consumption (%)', 'value': 'Renewable energy share in the total final energy consumption (%)'},
            {'label': 'CO2 Emissions (kt) by Country', 'value': 'CO2 Emissions (kt) by Country'},],
            value="Electricity Consumption in Malaysia",), 
    ]), 
    dbc.Row([
        dcc.Graph(id='CO2_plot', figure={}) 
    ]), 
], fluid = True)

# style = {'font-size': '15px', 
#                     'width': '90px', 'height': '30px', 
#                     'display': 'inline-block', 'justify-content': 'center', 
#                     'align-items': 'center'}, className="ml-4"), 