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
import style

from Stock_table import make_stock_table
from Stock_chart import make_stock_chart



Stock_page = html.Div([
    dcc.Store(id='table-memory'),
    dcc.Store(id='stock-memory'),
    html.Div([' '],style=dict(backgroundColor='#6155a6',
                             height = 25)),
    html.Div([' '],style=dict(backgroundColor='white',
                             height = 40)),
    html.Div([
        dcc.Tabs(id = 'Navigation',value='/Stock',
                children=[
                dcc.Tab(label='Overview',value ='/Overview',
                        style=style.NavTab,selected_style=style.NavTab_selected),
                dcc.Tab(label='Stock',value ='/Stock',
                        style=style.NavTab,selected_style=style.NavTab_selected),
                dcc.Tab(label='Fund',value ='/Fund',
                        style=style.NavTab,selected_style=style.NavTab_selected),
                dcc.Tab(label='Others',value ='/Others',
                        style=style.NavTab,selected_style=style.NavTab_selected),
                dcc.Tab(label='Invest',value ='/Invest-traceback',
                        style=style.NavTab,selected_style=style.NavTab_selected)
            ],style=dict(width='80%',margin='auto',height = 39))
    ],style=dict(backgroundColor='#f4f4f2',backgroundImage='linear-gradient(#FFFFFF 20% ,#f4f4f2 80%)',
                             height = 40,boxShadow='0px 15px 20px 0px #bbbfca')),
    html.Br(),
    html.Br(),
    html.Div([
        html.Div(id='stock-table-page',
                style=dict(border='0px solid #495464',
                                      width='35%',
                                      #height = 1000,
                                      display='inline-block',verticalAlign = 'top')),
        html.Div(id='stock-chart',
                style=dict(border='0px solid #495464',
                                      width='64.6%',
                                      #height = 1000,
                                      display='inline-block',verticalAlign = 'top')),
        ],style=dict(border='0px solid #495464',
                             height = 800,
                             width='85%',
                             margin='auto'),),
    html.Div([
        "Times Series Analysis"

        ],style=dict(border='0px solid #495464',
                             height = 600,
                             width='85%',
                             margin='auto'))

])
