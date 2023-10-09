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

global_df = pd.DataFrame()

dash.register_page(__name__, path='/')

result_margin = "15px"
label_width = '120px'
margin_btm = '5px'
input_width = '120px'
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Markdown(children = '**Inventory for optimization**'),
            html.Label("elec_target", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}), 
            dcc.Input(id='elec_requirement', min = 0, type='number', value=45813.43726, placeholder = "GWh/year", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("elec_nat", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}), 
            dcc.Input(id='elec_nat', type='number', min = 0, value=5094.566453, placeholder = "GWh/year/unit", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("elec_POME", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}), 
            dcc.Input(id='elec_POME', type='number', min = 0, value=45.42528437, placeholder = "GWh/year/unit",style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("elec_solar", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}), 
            dcc.Input(id='elec_solar', type='number', min = 0, value=73, placeholder = "GWh/year/unit", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),

        dbc.Col([
            dcc.Markdown(children = '**Cost Objective Function**'),
            html.Label("cost_nat", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}), 
            dcc.Input(id='cost_nat', type='number', min = 0, value=102763.798, placeholder = "RM/year/GWh",style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("cost_POME",style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}), 
            dcc.Input(id='cost_POME', type='number', min = 0, value=4053247.211, placeholder = "RM/year/GWh", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("cost_solar", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='cost_solar', type='number', min = 0, value=398401.826, placeholder = "RM/year/GWh", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),

        dbc.Col([
            dcc.Markdown(children = '**CO2 Objective Function**'),
            html.Label("co2_nat", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='co2_nat', type='number', min = 0, value=332.968, placeholder = "GWh/year/unit", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("co2_pome", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='co2_pome', type='number', min = 0, value=483.264, placeholder = "GWh/year/unit",style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("co2_solar", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='co2_solar', type='number', min = 0, value=27.747, placeholder = "GWh/year/unit", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),

        dbc.Col([
            dcc.Markdown(children = '**Cost Weight**'),
            html.Label("Natural gas", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='w_cost_nat', type='number', min = 0, value=1.15, placeholder = "-", style={'margin-bottom': margin_btm,'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("Biogas", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='w_cost_POME', type='number', min = 0, value=1, placeholder = "-", style={'margin-bottom':margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("Solar", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='w_cost_solar', type='number', min = 0, value=1.05, placeholder = "-", style={'margin-bottom':margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dcc.Markdown('**CO2 Weight**'),
            html.Label("Natural gas", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='w_co2_nat', type='number', min = 0, value=1.15, placeholder = "-", style={'margin-bottom': margin_btm,'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("Biogas", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='w_co2_POME', type='number', min = 0, value=1, placeholder = "-", style={'margin-bottom':margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("Solar", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='w_co2_solar', type='number', min = 0, value=1.2, placeholder = "-", style={'margin-bottom':margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),

        dbc.Col([
            dcc.Markdown(children = '**Biomass Availability Constraints**'),
            html.Label("POME_max", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='POME_max', type='number', min = 0, value=8839760.915, placeholder = "ton/year", style={'margin-bottom': margin_btm,'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("POME_each", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='POME_each', type='number', min = 0, value=331444.08*1.10231, placeholder = "ton/year", style={'margin-bottom':margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),

        dbc.Col([
            dcc.Markdown(children = '**Land Constraints**'),
            html.Label("available_land", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='solar_land_max', type='number', min = 0, value=2432978707, placeholder = "m2", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
            html.Label("land_solar", style={'margin-bottom': margin_btm, 'width': label_width, 'display': 'inline-block'}),
            dcc.Input(id='solar_panel_each', type='number', min = 0, value=120000, placeholder = "m2", style={'margin-bottom': margin_btm, 'width': input_width, 'display': 'inline-block', 'text-align': 'center'}),
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),

        dbc.Col([
            html.Button('Optimize!', id='optimize', 
                    n_clicks=0, style = {'font-size': '15px', 
                    'width': '90px', 'height': '30px', 
                    'display': 'inline-block', 'justify-content': 'center', 
                    'align-items': 'center'}, className="ml-4"), 
            html.Div(id='dfframe', style={'display': 'none'}), 
            dbc.Spinner(html.Div(id="loading-output")),
            html.Div(id = 'status', children = "Ready to optimize result...", style = {'margin-bottom': "15px", 'margin-top': "15px"}), 
            html.Button("Download Result", id="btn_download_result", className="ml-4"),
            dcc.Download(id="download-df")

        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(id='dummy')
            ], style={'margin-bottom': result_margin}),
            dbc.Row([
               dcc.Markdown('**Continuous Result**')
            ]), 
            dbc.Row([
                html.Div(id='weight_container')
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x1")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x2")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x3")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "f1")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "f2")
            ]), 
            dbc.Row([
                html.Div(id='dummy2')
            ], style={'margin-bottom': result_margin}),
            dbc.Row([
               dcc.Markdown('**Integer Result**')
            ]), 
            dbc.Row([
                html.Div(id = "x1_int")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x2_int")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x3_int")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "f1_int")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "f2_int")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id='dummy3')
            ], style={'margin-bottom': result_margin}),
            dbc.Row([
               dcc.Markdown('**Share of Energy**')
            ]), 
            dbc.Row([
                html.Div(id = "x1_share")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x2_share")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "x3_share")
            ], style={'margin-bottom': result_margin}),
            dbc.Row([
                html.Div(id='dummy4')
            ], style={'margin-bottom': result_margin}),
            
            dbc.Row([
               dcc.Markdown('**Constraints**')
            ]), 
            dbc.Row([
                html.Div(id = "total_generation")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "pome_consumption")
            ], style={'margin-bottom': result_margin}), 
            dbc.Row([
                html.Div(id = "solar_consumption")
            ], style={'margin-bottom': result_margin}), 
        ], width=3, style = {'text-align':'center', 'display': 'inline-block', 'text-align': 'center'}), 
        
        dbc.Col([
            dbc.Row([
                html.Div([
                    dcc.Slider(0, 1, 0.05, 
                    value = 0.5, 
                    id = 'weight_slider', 
                    className = "slider")
                ])
            ]), 
            dbc.Row([
                dcc.Dropdown(id='design space-dropdown', 
                     options=[
                    {'label': 'Design Space Integer', 'value': 'Design Space Integer'},
                    {'label': 'Design Space Continuous', 'value': 'Design Space Continuous'},
                    {'label': 'Objective Space Integer', 'value': 'Objective Space Integer'},
                    {'label': 'Objective Space Continuous', 'value': 'Objective Space Continuous'}],
                     value="Objective Space Continuous",), 
            ]), 
            dbc.Row([
                dcc.Graph(id='objective_space', figure={}) 
            ])
        ], width=8)

    ]),
     
], fluid = True)
