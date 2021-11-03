import dash
import dash_core_components as dcc
import dash_html_components as html
from DatabaseManagement import *
from pandas_datareader import data

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import seaborn  as sns
import os
from dash.dependencies import Input, Output
sns.set_style('darkgrid')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

#import Stock_page
from Stock_page import Stock_page

app = dash.Dash()

app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    dcc.Location(id = 'url',refresh=False),

    html.Div(id = 'page-content'),
])
@app.callback(
    Output('page-content','children'),
    [Input('url','pathname')]
)
def display_page(pathname):
    if pathname =='/Stock':
        return Stock_page
    elif pathname == '/Fund':
        return html.Div('Fund')
    elif pathname == '/Overview':
        return html.Div('Overview')
    elif pathname == '/Others':
        return html.Div('Overview')
    elif pathname == '/Invest-traceback':
        return html.Div('Invest-traceback')
    else:
        return html.Div('404')

from stockpage_callbacks import *


if __name__ == '__main__':
    app.run_server(debug=True)
