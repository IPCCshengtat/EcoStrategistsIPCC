import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import opt3
import dash as dash
from dash import dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pymoo.mcdm.pseudo_weights import PseudoWeights
import time
import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real
import math
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

# For Data Visualization
import plotly as py
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.offline as pyo

from plotly.subplots import make_subplots
from plotly.offline import plot

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


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], use_pages=True)
server = app.server

app.layout = dbc.Container([

    html.Div([
        html.H3('Optimizing the Share of Resources for Electricity Production in Johor, Malaysia alongside Recurrent Neural Network (RNN) time series forecasting - EcoStrategists', style={'textAlign': 'center'})
    ]), 
    html.Div([
        dcc.Link(page['name']+"         |         ", href=page['path'])
        for page in dash.page_registry.values()
    ]),
    html.Hr(),

    # content of each page
    dash.page_container
])


@app.callback(
    dash.dependencies.Output("loading-output", "children"), 
    [dash.dependencies.Input("optimize", "n_clicks"),], 
)
def load_output(n):
    if n:
        return f"Optimizing"
    return ""

@app.callback(
    dash.dependencies.Output("loading-output", "children", allow_duplicate=True), 
    [dash.dependencies.Input("optimize", "n_clicks"),], prevent_initial_call='initial_duplicate'
)
def load_output(n):
    if n:
        time.sleep(1)
        return ""
    return ""

@app.callback(
    [dash.dependencies.Output('status', 'children'),
     dash.dependencies.Output('dfframe', 'children'),],
    [dash.dependencies.Input(component_id='optimize', component_property='n_clicks')], 
    [dash.dependencies.State('elec_requirement', 'value'),
    dash.dependencies.State('elec_nat', 'value'),
    dash.dependencies.State('elec_POME', 'value'),
    dash.dependencies.State('elec_solar', 'value'),
    dash.dependencies.State('cost_nat', 'value'),
    dash.dependencies.State('cost_POME', 'value'),
    dash.dependencies.State('cost_solar', 'value'),
    dash.dependencies.State('co2_nat', 'value'),
    dash.dependencies.State('co2_pome', 'value'),
    dash.dependencies.State('co2_solar', 'value'),
    dash.dependencies.State('solar_land_max', 'value'),
    dash.dependencies.State('solar_panel_each', 'value'),
    dash.dependencies.State('POME_max', 'value'),
    dash.dependencies.State('POME_each', 'value'),
    dash.dependencies.State('w_cost_nat', 'value'),
    dash.dependencies.State('w_cost_POME', 'value'),
    dash.dependencies.State('w_cost_solar', 'value'),
    dash.dependencies.State('w_co2_nat', 'value'),
    dash.dependencies.State('w_co2_POME', 'value'),
    dash.dependencies.State('w_co2_solar', 'value'),
    dash.dependencies.State('status', 'children'),
    ], prevent_initial_call=True)
def update_graph(n_clicks, elec_requirement, 
                 elec_nat, elec_POME, elec_solar, cost_nat, cost_POME, 
                 cost_solar, co2_nat, co2_pome, co2_solar, solar_land_max, 
                 solar_panel_each, POME_max, POME_each, w_cost_nat, w_cost_POME, 
                 w_cost_solar, w_co2_nat, w_co2_POME, w_co2_solar, output_text):    
    print("Running the calculatiosn")
    if n_clicks:
        resultdf = opt3.runopt(elec_requirement, elec_nat, elec_POME, elec_solar, cost_nat, cost_POME, cost_solar, co2_nat, co2_pome, co2_solar, solar_land_max, solar_panel_each, POME_max, POME_each, w_cost_nat, w_cost_POME, w_cost_solar, w_co2_nat, w_co2_POME, w_co2_solar)
        global global_df
        global_df = resultdf
        global solar_panel_each_check
        global POME_each_check
        solar_panel_each_check = solar_panel_each
        POME_each_check = POME_each

        table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in resultdf.columns],
        data=resultdf.to_dict('records')
        )
        resultdf.to_excel("Optimized Result.xlsx")
        return "Optimized Result Ready !!", table
    return output_text, None

@app.callback(
    [dash.dependencies.Output('objective_space', 'figure'), 
    dash.dependencies.Output('weight_container', 'children'),
    dash.dependencies.Output('x1', 'children'), 
    dash.dependencies.Output('x2', 'children'), 
    dash.dependencies.Output('x3', 'children'), 
    dash.dependencies.Output('f1', 'children'), 
    dash.dependencies.Output('f2', 'children'), 
    dash.dependencies.Output('x1_int', 'children'), 
    dash.dependencies.Output('x2_int', 'children'), 
    dash.dependencies.Output('x3_int', 'children'), 
    dash.dependencies.Output('f1_int', 'children'), 
    dash.dependencies.Output('f2_int', 'children'), 
    dash.dependencies.Output('total_generation', 'children'),
    dash.dependencies.Output('pome_consumption', 'children'),
    dash.dependencies.Output('solar_consumption', 'children'),
    dash.dependencies.Output('x1_share', 'children'),
    dash.dependencies.Output('x2_share', 'children'),
    dash.dependencies.Output('x3_share', 'children'),], 
    [dash.dependencies.Input('dfframe', 'children'), 
    dash.dependencies.Input('weight_slider', 'value'), 
    dash.dependencies.Input("design space-dropdown", "value")], prevent_initial_call = True)
def plot_objective_space(dataframe, value, chosenvalue):
    template_plotly = "plotly_dark"
    resultdf = global_df
    category_colors = {
        'Dominant': 'rgb(0, 255, 0)',  # Red
        'Non-Dominant': 'rgb(250, 2, 130)',  # Green
    }
    category_opacity = {
        'Dominant': 0.8,  # Red
        'Non-Dominant': 0.2,  # Green
    }
    # category_name = {
    #     'Dominant': "Dominant",
    #     'Non-Dominant': "Non-Dominant",
    # }
    if "Integer" in chosenvalue:
        dfFn = global_df[["Dominance", "f1_cal_int_norm", "f2_cal_int_norm"]]
        dfFn = dfFn.sort_values("f1_cal_int_norm")
        dfFn = dfFn[dfFn["Dominance"] != "Non-Dominant"].drop(columns=["Dominance"], axis = 1).to_numpy().reshape(-1, 2)
        weights = np.array([(1-value), value])
        i = PseudoWeights(weights).do(dfFn)
        row_index = global_df.index[global_df['f1_cal_int_norm'] == dfFn[i, 0]].tolist()[0]
        resultseries = global_df.iloc[row_index, :]
        
        fig = go.Figure()
        if "Objective" in chosenvalue:
        # Add a scatter trace for each category
            for category, color in category_colors.items():
                category_df = resultdf[resultdf['Dominance'] == category]
                fig.add_trace(go.Scatter(
                    x=category_df['f1_calc_int'],
                    y=category_df['f2_calc_int'],
                    mode='markers',
                    marker=dict(
                        color=color,
                        opacity = category_opacity[category], 
                        size = 13, 
                    ),
                    name=category,
                ))
            fig.update_layout(
                xaxis=dict(title='Cost Obj (RM/year)'),
                yaxis=dict(title='CO2 Obj (Ton/year)'),
                title='Objective Space',
                showlegend=True,
                template = template_plotly, 
            )
            pareto_df = global_df[global_df["Dominance"] == "Dominant"].sort_values(by = "f1_calc_int")

            fig.add_trace(go.Scatter(x= pd.Series(resultseries["f1_calc_int"]), y=pd.Series(resultseries["f2_calc_int"]), marker=dict(size=16, symbol = 'x'), name='Weighted Solutions'))
            fig.add_trace(go.Line(x = pareto_df["f1_calc_int"], y = pareto_df["f2_calc_int"], name='Pareto Front'))
        elif "Design" in chosenvalue:
            for category, color in category_colors.items():
                category_df = resultdf[resultdf['Dominance'] == category]
                fig.add_trace(go.Scatter3d(
                    x=category_df['x1 (int)'],
                    y=category_df['x2 (int)'],
                    z=category_df['x3 (int)'],
                    mode='markers',
                    marker=dict(
                        color=color,
                        opacity = category_opacity[category], 
                        size = 8, 
                    ),
                    name=category,
                ))
            fig.update_layout(
                scene = dict(
                xaxis = dict(title = 'GWh from Nat-Gas'),
                yaxis = dict(title = "GWh from POME"), 
                zaxis = dict(title = "GWh from Solar")
                ),
                title = "Design Space", 
                showlegend = True, 
                template = template_plotly, 
            )
            fig.add_trace(go.Scatter3d(x = pd.Series(resultseries["x1 (int)"]),y = pd.Series(resultseries["x2 (int)"]), z = pd.Series(resultseries["x3 (int)"]), marker=dict(size=10, symbol = 'x'), name='Weighted Solutions'))


    elif "Continuous" in chosenvalue:
        dfFn = global_df[["Dominance_Cont", "f1_pymoo_norm", "f2_pymoo_norm"]]
        dfFn = dfFn.sort_values("f1_pymoo_norm")
        dfFn = dfFn[dfFn["Dominance_Cont"] != "Non-Dominant"].drop(columns=["Dominance_Cont"], axis = 1).to_numpy().reshape(-1, 2)
        weights = np.array([(1-value), value])
        i = PseudoWeights(weights).do(dfFn)
        row_index = global_df.index[global_df['f1_pymoo_norm'] == dfFn[i, 0]].tolist()[0]
        resultseries = global_df.iloc[row_index, :]

        resultdf_ordered = resultdf.sort_values(by = "f1_pymoo")
        fig = go.Figure()
        if "Objective" in chosenvalue:
        # Add a scatter trace for each category
            for category, color in category_colors.items():
                category_df = resultdf[resultdf['Dominance_Cont'] == category]
                fig.add_trace(go.Scatter(
                    x=category_df['f1_pymoo'],
                    y=category_df['f2_pymoo'],
                    mode='markers',
                    marker=dict(
                        color=color,
                        opacity = category_opacity[category], 
                        size = 13, 
                    ),
                    name=category,
                ))

            fig.update_layout(
                xaxis=dict(title='Cost Obj (RM/year)'),
                yaxis=dict(title='CO2 Obj (Ton/year)'),
                title='Objective Space',
                showlegend=True,
                template = template_plotly, 
            )
            pareto_df = global_df[global_df["Dominance_Cont"] == "Dominant"].sort_values(by = "f1_pymoo")

            fig.add_trace(go.Scatter(x= pd.Series(resultseries["f1_pymoo"]), y=pd.Series(resultseries["f2_pymoo"]), marker=dict(size=16, symbol = 'x'), name='Weighted Solutions'))
            fig.add_trace(go.Line(x = pareto_df["f1_pymoo"], y = pareto_df["f2_pymoo"], name='Pareto Front'))
        elif "Design" in chosenvalue:
            for category, color in category_colors.items():
                category_df = resultdf[resultdf['Dominance_Cont'] == category]
                fig.add_trace(go.Scatter3d(
                    x=category_df['x1'],
                    y=category_df['x2'],
                    z=category_df['x3'],
                    mode='markers',
                    marker=dict(
                        color=color,
                        opacity = category_opacity[category], 
                        size = 8, 
                    ),
                    name=category,
                ))
            fig.update_layout(
                scene = dict(
                xaxis = dict(title = 'GWh from Nat-Gas'),
                yaxis = dict(title = "GWh from POME"), 
                zaxis = dict(title = "GWh from Solar")
                ),
                title = "Design Space", 
                showlegend = True, 
                template = template_plotly, 
            )
            fig.add_trace(go.Scatter3d(x = pd.Series(resultseries["x1"]),y = pd.Series(resultseries["x2"]), z = pd.Series(resultseries["x3"]), marker=dict(size=10, symbol = 'x'), name='Weighted Solutions'))

    fig.update_layout(
            width=1100,
            height=700
        )
    
    
    return fig, f'Weight of CO2: {value}', f'Natural Gas Share: {np.format_float_scientific(resultseries["x1"], 4)} GWh/yr', f'Biomass Share: {np.format_float_scientific(resultseries["x2"], 4)} GWh/yr', f'Solar Share: {np.format_float_scientific(resultseries["x3"], 4)} GWh/yr', f'Cost Required: {np.format_float_scientific(resultseries["f1_pymoo"], 4)} RM/yr', f'CO2 emission: {np.format_float_scientific(resultseries["f2_pymoo"], 4)} ton/yr', f'(Int) Natural Gas Share: {resultseries["x1 (int)"]} units', f'(Int) Biomass Share: {resultseries["x2 (int)"]} units', f'(Int) Solar Share: {resultseries["x3 (int)"]} units', f'(Int) Cost Required: {np.format_float_scientific(resultseries["f1_calc_int"], 4)} RM/yr', f'(Int) CO2 emission: {np.format_float_scientific(resultseries["f2_calc_int"], 4)} ton/yr', f'Total Generation: {np.format_float_scientific(resultseries["x1"] + resultseries["x2"] + resultseries["x3"], 4)} GWh/yr', f'POME Consumption: {np.format_float_scientific(resultseries["x2 (int)"]*POME_each_check, 4)} ton/yr', f'Land Consumption: {np.format_float_scientific(resultseries["x3 (int)"]*solar_panel_each_check, 4)} m2', f'Natural Gas Share: {round(resultseries["x1"]/(resultseries["x1"] + resultseries["x2"] + resultseries["x3"]), 3)}', f'POME Share: {round(resultseries["x2"]/(resultseries["x1"] + resultseries["x2"] + resultseries["x3"]), 3)}', f'Solar Share: {round(resultseries["x3"]/(resultseries["x1"] + resultseries["x2"] + resultseries["x3"]), 3)}'

@app.callback(
    dash.dependencies.Output("download-df", "data"),
    dash.dependencies.Input("btn_download_result", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(global_df.to_csv, "Optimized Result.csv")


@app.callback(
    [dash.dependencies.Output('RNN_plot', 'figure')], 
    [dash.dependencies.Input("RNN-dropdown", "value"), 
    dash.dependencies.Input("resolution_slider", "value")])
def plot_RNN(dropdownvalue, resolutionslider):
    template_plotly = "plotly_dark"
    RNN_original_df = pd.read_excel("RNN_df_streamlit.xlsx", sheet_name="original")
    RNN_test_pred_df = pd.read_excel("RNN_df_streamlit.xlsx", sheet_name="result")
    fig = go.Figure()
    if dropdownvalue == "Original Dataset":
        print("Original Dataset")
        fig.add_trace(go.Scatter(x = RNN_original_df["Date"][::resolutionslider], 
                    y = RNN_original_df["Electricity"][::resolutionslider],
                    mode = "lines", 
                    marker = dict(
                    size = 5, )))
        fig.update_layout(
            xaxis=dict(title='Time'),
            yaxis=dict(title='Electricity Consumption (kWh)'),
            title='Original Dataset',
            template = template_plotly, )
    elif dropdownvalue == "Accuracy Plot":
        print("Training Loss")
        fig.add_trace(go.Scatter(x = RNN_test_pred_df["y_test"][::resolutionslider], 
                    y = RNN_test_pred_df["y_pred"][::resolutionslider],
                    mode = "markers",
                    name = "points",
                    marker = dict(
                    size = 5, )))
        fig.add_trace(go.Scatter(x = RNN_test_pred_df["y_test"][::resolutionslider], 
            y = RNN_test_pred_df["y_test"][::resolutionslider], 
            name = "y<sub>test</sub> vs y<sub>test</sub>", mode = "lines")
            )
        fig.update_layout(
            xaxis=dict(title='Y<sub>test</sub>'),
            yaxis=dict(title='Y<sub>predict</sub>'),
            title='Accuracy Plot',
            template = template_plotly, 
            showlegend=True,)
    elif dropdownvalue == "Result Plot":
        print("Result Plot")
        fig.add_trace(go.Scatter(x = RNN_original_df.loc[43824:, "Date"][::resolutionslider], 
                                 y = RNN_test_pred_df["y_test"][::resolutionslider], 
                                 name = "Testing points",
                                mode = "lines", 
                                marker = dict(
                                    size = 5, )))
        fig.add_trace(go.Scatter(x = RNN_original_df.loc[43824:, "Date"][::resolutionslider], 
                                 y = RNN_test_pred_df["y_pred"][::resolutionslider], 
                                 name = "Predict points",
                                mode = "lines", 
                                marker = dict(
                                    size = 5, )))
        fig.update_layout(
            xaxis=dict(title='2019 Year'),
            yaxis=dict(title='Electricity Consumption Prediciton (kWh)'),
            title='RNN Result Plot',
            template = template_plotly, 
            showlegend=True,)
    else: 
        fig = go.Figure()
        print("Nothing.....")

    return [fig]

@app.callback(
    [dash.dependencies.Output('Running', 'figure'), 
     dash.dependencies.Output('Running_2', 'figure'),], 
    [dash.dependencies.Input("resolution_slider", "value")])
def plot_RNN_running( resolutionslider):
    template_plotly = "plotly_dark"
    RNN_original_df = pd.read_excel("RNN_df_streamlit.xlsx", sheet_name="original")
    RNN_test_pred_df = pd.read_excel("RNN_df_streamlit.xlsx", sheet_name="result")
    x_axis =  list(RNN_original_df.loc[43824:, "Date"][::resolutionslider])
    y_axis1 = list(RNN_test_pred_df["y_test"][::resolutionslider])
    y_axis2 = list(RNN_test_pred_df["y_pred"][::resolutionslider])
    frames = []
    frames2 = []

    for frame in range(1, len(x_axis)+1):
        x_axis_frame = np.arange(frame)
        y_axis_1_frame = list(y_axis1[0:frame])
        y_axis_2_frame = list(y_axis2[0:frame])
        curr_frame = go.Frame(data = [go.Scatter(x = x_axis_frame, y = y_axis_1_frame, mode = "lines"), ])
        curr_frame_2 = go.Frame(data = [go.Scatter(x = x_axis_frame, y = y_axis_2_frame, mode = "lines"), ])
        frames.append(curr_frame)
        frames2.append(curr_frame_2)

    fig = go.Figure(
        data = [go.Scatter(x = np.array([0]), y = np.array([0]), mode = "lines", showlegend=False)],
        layout=go.Layout(
            # xaxis=dict(range=[0, 5], autorange=False),
            yaxis=dict(range=[0, 400], autorange=False),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None,
                                dict(frame=dict(duration=30, redraw=True), fromcurrent=True,)]
                        )],
                    showactive=False,
                    direction='left',
                    x=0.0,
                    y=0.0,
                    name="animation-button-1" # set the name parameter to specify the ID of the button
                )]),
        frames = frames,
    )
    fig.update_layout(
        xaxis=dict(title='Iteration'),
        yaxis=dict(title='Electricity Consumption Prediciton (kWh)'),
        title='Ground Truth',
        template = template_plotly, 
        showlegend=True,)
    
    fig2 = go.Figure(
        data = [go.Scatter(x = np.array([0]), y = np.array([0]), mode = "lines", showlegend=False)],
        layout=go.Layout(
            # xaxis=dict(range=[0, 5], autorange=False),
            yaxis=dict(range=[0, 400], autorange=False), 
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None,
                                dict(frame=dict(duration=30, redraw=True), fromcurrent=True,)]
                        )],
                    showactive=False,
                    direction='left',
                    x=0.0,
                    y=0.0,
                    name="animation-button-2" # set the name parameter to specify the ID of the button
                )]),
        frames = frames2,
    )
    fig2.update_layout(
        xaxis=dict(title='Iteration'),
        yaxis=dict(title='Electricity Consumption Prediciton (kWh)'),
        title='Prediction',
        template = template_plotly, 
        showlegend=True,)

    return fig, fig2

@app.callback(
    [dash.dependencies.Output('ANN_plot', 'figure')], 
    [dash.dependencies.Input("ANN-dropdown", "value"), 
    dash.dependencies.Input("resolution_slider_ANN", "value")])
def plot_ANN(dropdownvalue, resolutionslider):
    template_plotly = "plotly_dark"
    ANN_original_df = pd.read_excel("train_protected.xlsx", sheet_name="before_trained")
    fig = go.Figure()
    if dropdownvalue == "Population Dataset":
        print("Population Dataset")
        fig.add_trace(go.Scatter(x = ANN_original_df["Population_noise"], 
                    y = ANN_original_df["Electricity Consumption_noise"],
                    mode = "markers",  
                    marker = dict(
                        size = 5, )))
        fig.update_layout(
            xaxis=dict(title='Population'),
            yaxis=dict(title='Electricity Consumption (GWh)'),
            title='Population Data',
            template = template_plotly, 
            showlegend=False,)
        
    elif dropdownvalue == "GDP Dataset":
        print("GDP Dataset")
        fig.add_trace(go.Scatter(x = ANN_original_df["GDP (RM mil)_noise"], 
                    y = ANN_original_df["Electricity Consumption_noise"],
                    mode = "markers",  
                    marker = dict(
                        size = 5, )))
        fig.update_layout(
            xaxis=dict(title='GDP (RM mil)'),
            yaxis=dict(title='Electricity Consumption (GWh)'),
            title='GDP Data',
            template = template_plotly, 
            showlegend=False,)

        
    elif dropdownvalue == "Accuracy Plot":
        print("Training Loss")
        fig.add_trace(go.Scatter(x = ANN_original_df.loc[-252:, "Electricity Consumption_noise"],
                y = ANN_original_df.loc[-252:, "Electricity Predict"],  
                name = "Data points",
                mode = "markers", 
                marker = dict(
                    size = 5, 
                    color = 'white'
                )))
        fig.add_trace(go.Scatter(x = ANN_original_df.loc[-252:, "Electricity Predict"], 
                        y = ANN_original_df.loc[-252:, "Electricity Predict"], 
                        name = "y<sub>test</sub> vs y<sub>test</sub>", 
                        mode = "lines", ))
        fig.update_layout(
            xaxis=dict(title='Y<sub>test</sub>'),
            yaxis=dict(title='Y<sub>predict</sub>'),
            title='Accuracy Plot',
            template = template_plotly, 
            showlegend=True,)
        
    elif dropdownvalue == "Result Plot":
       print("Result Plot")
       fig.add_trace(go.Scatter(x = ANN_original_df["Year"], 
                    y = ANN_original_df["Electricity Consumption_noise"],
                    mode = "markers",
                    name = "Interpolation",   
                    marker = dict(
                        size = 5, 
                        color = 'white')))
       fig.add_trace(go.Scatter(x = ANN_original_df.loc[-252:, "Year"], 
                    y = ANN_original_df.loc[-252:, "Electricity Predict"], 
                    name = "Prediction", 
                    mode = "markers", 
                    marker = dict(
                        size = 5, 
                        color = 'orange')))
       fig.update_layout(
            xaxis=dict(title='Year'),
            yaxis=dict(title='Electricity Consumption (GWh)'),
            title='ANN Result Plot',
            template = template_plotly, 
            showlegend=True,)
       
    else: 
        fig = go.Figure()
        print("Nothing.....")

    return [fig]

@app.callback(
    dash.dependencies.Output("CO2_plot", "figure"),
    dash.dependencies.Input("CO2_map", "value"), 
)
def geographs(choice):
    data = pd.read_csv(r'energy.csv')
    my = ['Malaysia']
    asia = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'Cyprus', 'Georgia', 'Hong Kong', 'Indonesia', 
    'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Macao', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 
    'North Korea', 'Oman', 'Pakistan', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'State of Palestine', 'Syria', 'Taiwan', 'Tajikistan', 
    'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen', 'China', 'India']
    data_my = data[data['Entity'].isin(my)]
    data_asia = data[data['Entity'].isin(asia)]

    # Assume new parameters
    electricity_my = data_my[['Entity','Year','FF (TWh)', 'Nuclear (TWh)','RE (TWh)',]]
    electricity_my_sum = electricity_my.groupby('Year').sum()
    electricity_my_sum['Total Electricity'] = electricity_my_sum['FF (TWh)'] + electricity_my_sum['Nuclear (TWh)'] + electricity_my_sum['RE (TWh)']
    electricity_my_sum['FF Share'] = (electricity_my_sum['FF (TWh)']/electricity_my_sum['Total Electricity'])*100
    electricity_my_sum['Nuclear Share'] = (electricity_my_sum['Nuclear (TWh)']/electricity_my_sum['Total Electricity'])*100
    electricity_my_sum['RE Share'] = (electricity_my_sum['RE (TWh)']/electricity_my_sum['Total Electricity'])*100
    fig = go.Figure()

    if choice == "Electricity Consumption in Malaysia":
        fig = fig.add_trace(go.Bar(x = electricity_my_sum.index, y = electricity_my_sum['FF Share'], name = 'Fossil Fuel'))
        fig = fig.add_trace(go.Bar(x = electricity_my_sum.index, y = electricity_my_sum['RE Share'], name = 'Renewable'))
        fig = fig.update_layout(
            title = 'Electricity Consumption in Malaysia', 
            xaxis_tickfont_size = 12, 
            xaxis_title="Years",
            yaxis_title="Electricity Comsumption (%)",
            template = 'plotly_dark', 
            barmode = 'stack')
    elif choice == "Renewable energy share in the total final energy consumption (%)" or choice == "CO2 Emissions (kt) by Country":
        column_name = choice
        for year in range(2000, 2020):
            # Filter the data for the current year
            filtered_data = data_asia[data_asia['Year'] == year]
            # Create a choropleth trace for the current year
            trace = go.Choropleth(
                locations=filtered_data['Entity'],
                z=filtered_data[column_name],
                locationmode='country names',
                colorscale='Electric',  # Use a different color scale for better contrast
                colorbar=dict(title=column_name),
                zmin=data_asia[column_name].min(),
                zmax=data_asia[column_name].max(),
                visible=False,  # Set the trace to invisible initially
            )
            # Add the trace to the figure
            fig.add_trace(trace)
        # Set the first trace to visible
        fig.data[0].visible = True
        # Create animation steps
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method='update',
                args=[{'visible': [False] * len(fig.data)},  # Set all traces to invisible
                        {'title_text': f'{column_name} Map - {2000 + i}', 'frame': {'duration': 1000, 'redraw': True}}],
                label=str(2000 + i)  # Set the label for each step
            )
            step['args'][0]['visible'][i] = True  # Set the current trace to visible
            steps.append(step)
        # Create the slider
        sliders = [dict(
            active=0,
            steps=steps,
            currentvalue={"prefix": "Year: ", "font": {"size": 14}},  # Increase font size for slider label
        )]

        # Update the layout of the figure with increased size and change the template
        fig.update_layout(
            title_text=f'{column_name}',  # Set the initial title
            title_font_size=24,  # Increase title font size
            title_x=0.5,  # Center the title
            geo=dict(
                scope="asia",
                showframe=True,
                showcoastlines=False,
                projection_type='equirectangular'
                
            ),
            sliders=sliders,
            height=600,  # Set the height of the figure in pixels
            # width=1000,  # Set the width of the figure in pixels
            font=dict(family='Arial', size=12),  # Customize font family and size for the whole figure
            margin=dict(t=80, l=50, r=50, b=50),  # Add margin for better layout spacing
            template = 'plotly_white',
        )
    else:
        pass
    return fig

if __name__ == "__main__":
    app.run(debug=True)