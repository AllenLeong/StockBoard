from dash.dependencies import Input, Output

import dash_core_components as dcc
import dash_html_components as html
from DatabaseManagement import *
from pandas_datareader import data
from facts import security_names
import pandas as pd
import numpy as np

from uilt import *
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_stock_chart(security):


    df = get_data(security)

    def up_or_down(x):
        return x>=0

    df['Close_dif'] = df.Close.diff()

    df['Volume_diff'] = df.Volume.diff()
    df['Volume_up'] = df.Volume_diff.apply(up_or_down)
    df.index = df.Date
    dff = df

    color_map = {'bg':"#f0f5f9",'increase':'#c060a1','decrease':'#28abb9','line':['#1c2b2d','#1f6f8b','#99a8b2','#e6d5b8']}

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing= 0.1,
                        specs=[
                            [{'type':'xy',"secondary_y": True}],
                            [{"secondary_y": True}],
                            ],

                       row_heights=[0.7,0.3]
            )

    ## Candle
    fig.add_candlestick(x=dff['Date'],
                        open=dff['Open'],
                        high=dff['High'],
                        low=dff['Low'],
                        close=dff['Close'],
                        name='candle',
                        increasing_line_color= '#c060a1', decreasing_line_color= '#28abb9',
                        row=1,col=1
                       )
    fig.add_scatter(x=dff["Date"],y=dff["Close"],mode="lines",line=dict(width=2,color='#583d72'),name="Close")
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=5).mean().values,mode="lines",line=dict(width=1.7,color='#1c2b2d'),name='MA5', row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=12).mean().values,mode="lines",line=dict(width=1.5,color='#1f6f8b'),name='MA12', row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=24).mean().values,mode="lines",line=dict(width=1.2,color='#99a8b2'),name='MA24', row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=60).mean().values,mode="lines",line=dict(width=1,color='#e6d5b8'),name='MA60', row=1,col=1)

    Volume_chart  = go.Bar(x=dff.Date,y=dff.Volume,
                           marker=dict(color='#f0f5f9',
                                        line=dict(color=['red' if i <0 else 'green' for i in dff.Volume_diff])),
                           showlegend=False
                          )
    fig.add_trace(Volume_chart,col=1,row=2)

    macd_dif = ema(dff.iloc[:,3].values,12) - ema(dff.iloc[:,3].values,26)
    macd_dea = ema(macd_dif,9)
    macd_hist = 2*(macd_dif-macd_dea)

    fig.add_traces([
            go.Scatter(y=macd_dif,x=dff.index,name='DIF', marker={'color':color_map['line'][-1]},showlegend=False,visible=False),
            go.Scatter(y=macd_dea,x=dff.index,name='DEA', marker={'color':color_map['line'][-2]},showlegend=False,visible=False)
        ],cols=1,rows=2)
    fig.add_trace(go.Bar(y=macd_hist,x=dff.index,
                        name='HIST',
                        marker=dict(color=['red' if i>=0 else 'green' for i in macd_hist]),
                        showlegend=False,visible=False)
                 ,col=1,row=2,secondary_y=True)

    rsi14 = get_rsi_timeseries(dff.Close,14)
    rsi28 = get_rsi_timeseries(dff.Close,24)
    rsi56 = get_rsi_timeseries(dff.Close,56)

    n = len(rsi14)
    fig.add_trace(
        go.Scatter(y=[80 for _ in range(n)] + [20 for _ in range(n)],x =rsi14.index.to_list() + rsi14.index[::-1].to_list(),
                      fill='toself',fillcolor=color_map['increase'],opacity=0.05,line=dict(width=0.5,dash='dot'),showlegend=False,visible=False),
        col=1,row=2
    )
    fig.add_trace(
            go.Scatter(y=[70 for _ in range(n)] + [30 for _ in range(n)],x =rsi14.index.to_list() + rsi14.index[::-1].to_list(),
                      fill='toself',fillcolor='blue',opacity=0.05,line=dict(width=0.5,dash='dot'),showlegend=False,visible=False),
        col=1,row=2
    )

    fig.add_traces([
            go.Scatter(y=rsi14,x=rsi14.index,name='RSI14',line=dict(color=color_map['line'][-1]),showlegend=False,visible=False),
            go.Scatter(y=rsi28,x=rsi28.index,name='RSI28',line=dict(color=color_map['line'][-2]),showlegend=False,visible=False),
            go.Scatter(y=rsi56,x=rsi56.index,name='RSI56',line=dict(color=color_map['line'][-3]),showlegend=False,visible=False)
        ],cols=1,rows=2)


    Hn = dff.High.rolling(9).max()
    Hn.fillna(value=dff.High.expanding().max(), inplace=True)
    Ln = dff.Low.rolling(9).min()
    Ln.fillna(value=dff.Low.expanding().min(), inplace=True)
    Cn = dff.Close
    rsv = (Cn-Ln)/(Hn-Ln)*100
    rsv.fillna(0)
    Kn = ema(rsv[9:],2)
    Dn = ema(Kn,2)
    Jn = 3*Kn - 2*Dn

    fig.add_traces([
         go.Scatter(x = rsv[9:].index,y=Dn,line=dict(shape='spline',color=color_map['line'][-2]),name="D:{: .2f}".format(Dn[-1]),showlegend=False,visible=False),
         go.Scatter(x = rsv[9:].index,y=Jn,line=dict(shape='spline',color=color_map['line'][-1]),name="J:{: .2f}".format(Jn[-1]),showlegend=False,visible=False),
         go.Scatter(x = rsv[9:].index,y=Kn,line=dict(shape='spline',color=color_map['line'][-3]),name="K:{: .2f}".format(Kn[-1]),showlegend=False ,visible=False),
     ],rows=2,cols=1)


    fig.add_trace(
            go.Scatter(y=[80 if i <80 else i for i in Kn] + [80 for _ in range(len(Kn))] ,x =rsv[9:].index.to_list() + rsv[9:].index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['increase'], opacity=0.4,line=dict(shape='spline'),showlegend=False,visible=False),
        row=2,col=1
    )

    fig.add_trace(
            go.Scatter(y=[20 if i >20 else i for i in Kn] + [20 for _ in range(len(Kn))] ,x =rsv[9:].index.to_list() + rsv[9:].index[::-1].to_list(),
                      fill='toself',fillcolor= color_map['decrease'],opacity=0.4,line=dict(shape='spline'),showlegend=False,visible=False),
        row=2,col=1
    )

    sub_chart = {'Volume':[0],'MACD':[1,2,3],'RSI':[4,5,6,7,8],'KDJ':[9,10,11,12,13]}
    visibility = [True,True,True,True,True,True]

    buttons = list([dict(label=key,
                     method="update",
                     args=[{"visible": visibility+[True if i in sub_chart[key] else False for i in range(27)]}])
               for key in sub_chart.keys()])





    fig.update_layout(
        margin = dict(l=50,r=0,t=60,b=0),
        height=600,
        #width=900,
        legend=dict(orientation = 'h',x=0.1,y=0.999,bgcolor = color_map['bg'],bordercolor=color_map['line'][0],borderwidth=1),
        font=dict(
            family="Times New Roman",
            size=18),
        hovermode = "x unified",
        xaxis=dict(

            rangeslider = dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=2,
                         label="2y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
        ),
        yaxis = dict(
         #fixedrange = False,autorange=True
        ),

         updatemenus=[
        dict(
            type="buttons",
            direction="right",
            #active=1,
            x=0.34,
            y=0.36,
            buttons= buttons,
            font=dict(size=14)
        )],

        plot_bgcolor=color_map['bg'],

        xaxis_tickformatstops = [
        dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
        dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),
        dict(dtickrange=[60000, 3600000], value="%H:%M m"),
        dict(dtickrange=[3600000, 86400000], value="%H:%M h"),
        dict(dtickrange=[86400000, 604800000], value="%e. %b d"),
        dict(dtickrange=[604800000, "M1"], value="%e. %b w"),
        dict(dtickrange=["M1", "M12"], value="%b '%y M"),
        dict(dtickrange=["M12", None], value="%Y Y")
    ]
    )
    display_days = 90
    fig.update_xaxes(showline=True, linewidth=1, linecolor=color_map['line'][2], mirror=True,
                     range = [df.Date.values[-display_days], df.Date.values[-1]], )
    fig.update_yaxes(showline=True, linewidth=1, linecolor=color_map['line'][2],mirror=True,row=1, col=1,
                     range = [df.Low.values[-display_days:].min(),df.High.values[-display_days:].max()*1.03],
                     showspikes=True, spikecolor="black", spikethickness=2)





    return html.Div([
        html.Div([security],style =dict(fontSize=40,marginLeft=50,fontFamily='Arial Black',color='#6155a6')),

        html.Div([securities[security]],style =dict(fontSize=16,marginLeft=50,fontFamily='Arial',color='#606470')),
        html.Div([security_names[security]],style =dict(fontSize=16,marginLeft=50,fontFamily='Arial',color='#606470')),
        dcc.Graph(figure=fig)])
