from StockBoard import app
from dash.dependencies import Input, Output

import dash_core_components as dcc
import dash_html_components as html
from DatabaseManagement import *
from pandas_datareader import data

import pandas as pd
import numpy as np

from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def update_fig(value):

    df = update_data('SZC')

    def up_or_down(x):
        return x>=0

    df['Close_dif'] = df.Close.diff()

    df['Volume_diff'] = df.Volume.diff()
    df['Volume_up'] = df.Volume_diff.apply(up_or_down)

    dff = df.iloc[-600:,]

    def ema(x,n,initial=False):
        output = np.zeros(len(x)).tolist()
        if initial:
            output[0] = initial
        else:
            output[0] = x[:n].mean()
        multiplier = 2/(n+1)
        for i in range(1,len(x)):
            output[i] = x[i]*multiplier  + output[i-1]*(1-multiplier)
        return np.array(output)

    macd_dif = ema(dff.iloc[:,3].values,12) - ema(dff.iloc[:,3].values,26)
    macd_dea = ema(macd_dif,9)
    macd_hist = 2*(macd_dif-macd_dea)


    def get_rsi_timeseries(prices, n=14):
        deltas = (prices-prices.shift(1)).fillna(0)
        avg_of_gains = deltas[1:n+1][deltas > 0].sum() / n
        avg_of_losses = -deltas[1:n+1][deltas < 0].sum() / n

        rsi_series = pd.Series(0.0, deltas.index)

        up = lambda x: x if x > 0 else 0
        down = lambda x: -x if x < 0 else 0
        i = n+1
        for d in deltas[n+1:]:
            avg_of_gains = ((avg_of_gains * (n-1)) + up(d)) / n
            avg_of_losses = ((avg_of_losses * (n-1)) + down(d)) / n
            if avg_of_losses != 0:
                rs = avg_of_gains / avg_of_losses
                rsi_series[i] = 100 - (100 / (1 + rs))
            else:
                rsi_series[i] = 100
            i += 1

        return rsi_series


    rsi14 = get_rsi_timeseries(dff.Close,14)
    rsi28 = get_rsi_timeseries(dff.Close,24)
    rsi56 = get_rsi_timeseries(dff.Close,56)


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


    def wr(dff,n):
        Hn = dff.High.rolling(n).max()
        Hn.fillna(value=dff.High.expanding().max(), inplace=True)
        Ln = dff.Low.rolling(n).min()
        Ln.fillna(value=dff.Low.expanding().min(), inplace=True)
        Cn = dff.Close

        return (Hn-Cn)/(Hn-Ln)*100

    wr6 = wr(dff,6)
    wr10 = wr(dff,10)



    bias6 = (dff.Close/dff.Close.rolling(6).mean()-1)*100
    bias12 = (dff.Close/dff.Close.rolling(12).mean()-1)*100
    bias24 = (dff.Close/dff.Close.rolling(24).mean() -1)*100


    def obv_volume(x):
        if x>0: return 1
        elif x==0: return 0
        else: return -1
    def obv(dff):
        obv = [0]
        for i in dff.dif.apply(obv_volume)*dff.Volume:
            obv.append(obv[-1]+i)
        return pd.Series(data=np.array(obv[1:]),index=dff.index)

    OBV = obv(dff)
    maobv = OBV.rolling(60).mean()
    OBV_rate = OBV.diff()/OBV




    n = 20
    TP = (dff.High + dff.Low+ dff.Close)/3

    MA = TP.rolling(n).mean()
    MD = (MA-dff.Close).rolling(n).std()
    cci = (TP-MA)/(MD*0.015)



    def ROC(dff,n):
        return (dff.Close - dff.Close.shift(n))/dff.Close.shift(n)*100
    roc = ROC(dff,14)

    def add_text(x,y,name,number,bgcolor,color='white'):
        fig.add_annotation(x=x,y=y, xref="paper", yref="paper",showarrow=False,
                       text='{}: {:.2f}'.format(name,number),
                       font=dict(
                            family="Arial Black",
                            size=8,
                            color= color
                            ),
                       align="center",
                       bgcolor=bgcolor#(color_map['increase'] if dff.dif.values[-1]>0 else color_map['decrease'])
                        )


    color_map = {'bg':"#f0f5f9",'increase':'#c060a1','decrease':'#28abb9','line':['#1c2b2d','#1f6f8b','#99a8b2','#e6d5b8']}
    fig = make_subplots(rows=10, cols=1,
                        shared_xaxes=True,
                        vertical_spacing= 0.015,
                        specs=[
                            [{'type':'candlestick'}],
                            [{"secondary_y": True}],
                            [{"secondary_y":True}],
                            [{'type':'xy'}],
                            [{'type':'xy'}],
                            [{'type':'xy'}],
                            [{'type':'xy'}],
                            [{'type':'xy'}],
                            [{'type':'xy'}],
                            [{'type':'xy'}],
                            ],

                       row_heights=[0.3]+[(1-0.3)/9]*9
            )

    ## Candle
    fig.add_candlestick(x=dff['Date'],
                        open=dff['Open'],
                        high=dff['High'],
                        low=dff['Low'],
                        close=dff['Close'],
                        name='candle',
                        increasing_line_color= '#c060a1', decreasing_line_color= '#28abb9',
                       row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff["Close"],mode="lines",line=dict(width=2,color='#583d72'),name="Close")
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=5).mean().values,mode="lines",line=dict(width=1.7,color='#1c2b2d'),name='MA5', row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=12).mean().values,mode="lines",line=dict(width=1.5,color='#1f6f8b'),name='MA12', row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=24).mean().values,mode="lines",line=dict(width=1.2,color='#99a8b2'),name='MA24', row=1,col=1)
    fig.add_scatter(x=dff["Date"],y=dff.iloc[:,3].rolling(window=60).mean().values,mode="lines",line=dict(width=1,color='#e6d5b8'),name='MA60', row=1,col=1)

    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=11),
        height=1000,
        width = 1000,
        xaxis=dict(
            rangeslider = dict(visible=False),
            rangeselector=dict(
                buttons=list([
                    dict(count=5,
                         label="1w",
                         step="day",
                         stepmode="backward"),
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
        ),
        yaxis = dict(
         fixedrange = False,autorange=True
        )
    )

    ## ROC
    fig.add_traces([
            go.Scatter(y=[10 if i <10 else i for i in roc] + [10 for _ in range(len(roc))] ,x =roc.index.to_list() + cci.index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['increase'], opacity=0.2,line=dict(shape='spline'),showlegend=False),
            go.Scatter(y=[-10 if i >-10 else i for i in roc] + [-10 for _ in range(len(roc))] ,x =roc.index.to_list() + cci.index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['decrease'], opacity=0.2,line=dict(width=0.5,shape='spline'),showlegend=False),
            go.Scatter(x = roc.index,y=roc,line=dict( color=color_map['line'][-3]),name="roc:{:.2f}".format(roc[-1]),showlegend=False),
          ],cols=1,rows=2)

    ## macd
    fig.add_traces([
        go.Scatter(y=macd_dif,x=dff.index,name='DIF', marker={'color':color_map['line'][-1]}),
        go.Scatter(y=macd_dea,x=dff.index,name='DEA', marker={'color':color_map['line'][-2]})
        ],cols=1,rows=3)
    fig.update_yaxes(visible=False,secondary_y=True,col=1,row=3)
    fig.add_bar(y=macd_hist,x=dff.index,
                name='HIST',secondary_y=True,
                marker=dict(color=[color_map['increase'] if i>=0 else color_map['decrease'] for i in macd_hist]),
                col=1,row=3)

    ## RSI
    n = len(rsi14)
    fig.add_trace(
        go.Scatter(y=[80 for _ in range(n)] + [20 for _ in range(n)],x =rsi14.index.to_list() + rsi14.index[::-1].to_list(),
                      fill='toself',fillcolor=color_map['increase'],opacity=0.05,line=dict(width=0.5,dash='dot'),showlegend=False),
        col=1,row=4
    )
    fig.add_trace(
            go.Scatter(y=[70 for _ in range(n)] + [30 for _ in range(n)],x =rsi14.index.to_list() + rsi14.index[::-1].to_list(),
                      fill='toself',fillcolor='blue',opacity=0.05,line=dict(width=0.5,dash='dot'),showlegend=False),
        col=1,row=4
    )

    fig.add_traces([
            go.Scatter(y=rsi14,x=rsi14.index,name='RSI14',line=dict(color=color_map['line'][-1])),
            go.Scatter(y=rsi28,x=rsi28.index,name='RSI28',line=dict(color=color_map['line'][-2])),
            go.Scatter(y=rsi56,x=rsi56.index,name='RSI56',line=dict(color=color_map['line'][-3]))
        ],cols=1,rows=4)


    ## KDJ
    fig.add_traces([
         go.Scatter(x = rsv[9:].index,y=Dn,line=dict(shape='spline',color=color_map['line'][-2]),name="D:{: .2f}".format(Dn[-1])),
         go.Scatter(x = rsv[9:].index,y=Jn,line=dict(shape='spline',color=color_map['line'][-1]),name="J:{: .2f}".format(Jn[-1])),
         go.Scatter(x = rsv[9:].index,y=Kn,line=dict(shape='spline',color=color_map['line'][-3]),name="K:{: .2f}".format(Kn[-1])),
     ],rows=5,cols=1)


    fig.add_trace(
            go.Scatter(y=[80 if i <80 else i for i in Kn] + [80 for _ in range(len(Kn))] ,x =rsv[9:].index.to_list() + rsv[9:].index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['increase'], opacity=0.4,line=dict(shape='spline'),showlegend=False),
        row=5,col=1
    )

    fig.add_trace(
            go.Scatter(y=[20 if i >20 else i for i in Kn] + [20 for _ in range(len(Kn))] ,x =rsv[9:].index.to_list() + rsv[9:].index[::-1].to_list(),
                      fill='toself',fillcolor= color_map['decrease'],opacity=0.4,line=dict(shape='spline'),showlegend=False),
        row=5,col=1
    )

    ## W%R
    fig.add_traces([
         go.Scatter(x = wr6.index,y=wr6,line=dict( color=color_map['line'][-3]),name="wr6:{: .2f}".format(wr6[-1])),
         go.Scatter(x = wr10.index,y=wr10,line=dict( color=color_map['line'][-1]),name="wr10:{: .2f}".format(wr10[-1])),
     ],rows=6,cols=1)


    ## BIAS
    fig.add_traces([
         go.Scatter(x = bias6.index,y=bias6,line=dict( color=color_map['line'][-3]),name="bias6:{: .2f}".format(bias6[-1])),
         go.Scatter(x = bias12.index,y=bias12,line=dict( color=color_map['line'][-2]),name="bias12:{: .2f}".format(bias12[-1])),
         go.Scatter(x = bias24.index,y=bias24,line=dict( color=color_map['line'][-1]),name="bias24:{: .2f}".format(bias24[-1])),
     ],rows=7,cols=1)


    ## OBV
    fig.add_traces([
         go.Scatter(x = OBV.index,y=OBV,line=dict( color=color_map['line'][-3]),name="OBV:{:.2f}M".format(OBV[-1]/1000000)),
         go.Scatter(x = maobv.index,y=maobv,line=dict( color=color_map['line'][-1]),name="MAOBV:{:.2f}M".format(maobv[-1]/1000000))
     ],rows=8,cols=1)


    ##CCI
    fig.add_traces([
            go.Scatter(y=[100 if i <100 else i for i in cci] + [100 for _ in range(len(cci))] ,x =cci.index.to_list() + cci.index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['increase'], opacity=0.2,line=dict(shape='spline'),showlegend=False),
            go.Scatter(y=[-100 if i >-100 else i for i in cci] + [-100 for _ in range(len(cci))] ,x =cci.index.to_list() + cci.index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['decrease'], opacity=0.2,line=dict(shape='spline'),showlegend=False),
            go.Scatter(x = cci.index,y=cci,line=dict( color=color_map['line'][0]),name="cci:{:.2f}".format(cci[-1])),
          ],rows=9,cols=1)

    ## ROC

    fig.add_traces([
            go.Scatter(y=[10 if i <10 else i for i in roc] + [10 for _ in range(len(roc))] ,x =roc.index.to_list() + cci.index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['increase'], opacity=0.2,line=dict(shape='spline'),showlegend=False),
            go.Scatter(y=[-10 if i >-10 else i for i in roc] + [-10 for _ in range(len(roc))] ,x =roc.index.to_list() + cci.index[::-1].to_list(),
                      fill='toself', fillcolor= color_map['decrease'], opacity=0.2,line=dict(width=0.5,shape='spline'),showlegend=False),
            go.Scatter(x = roc.index,y=roc,line=dict( color=color_map['line'][-3]),name="roc:{:.2f}".format(roc[-1])),
          ],cols=1,rows=10)

    ma5 = dff.Close.values[-5:].mean()
    ma12 = dff.Close.values[-12:].mean()
    ma24 = dff.Close.values[-24:].mean()
    ma60 = dff.Close.values[-60:].mean()
    price = dff.Close.values[-1]
    prices = pd.DataFrame([[ma5,ma12,ma24,ma60],['MA5','MA12','MA24','MA60']]).T
    prices = prices.sort_values(0,ascending=False)

    for i in range(4):
        add_text(1.04,0.98-i*0.04,prices.iloc[i,1],prices.iloc[i,0],bgcolor=color_map['line'][prices.index[i]],color='white')



    add_text(1.04,0.705,'ROC14 ',roc.values[-1],bgcolor=color_map['line'][1],color='white')

    if macd_dif[-1]>macd_dea[-1]:
        add_text(1.04,0.628,'DIF ',macd_dif[-1],bgcolor=color_map['line'][-1],color='white')
        add_text(1.04,0.605,'DEA ',macd_dea[-1],bgcolor=color_map['line'][-2],color='white')
        add_text(1.04,0.582,'HIST ',macd_hist[-1],bgcolor=color_map['increase'],color='white')
    else:
        add_text(1.04,0.628,'DEA ',macd_dea[-1],bgcolor=color_map['line'][-1],color='white')
        add_text(1.04,0.605,'DIF ',macd_dif[-1],bgcolor=color_map['line'][-2],color='white')
        add_text(1.04,0.582,'HIST ',macd_hist[-1],bgcolor=color_map['decrease'],color='white')

    rsis = pd.DataFrame([[rsi14[-1],rsi28[-1],rsi56[-1]],['RSI14','RSI28','RSI56']]).T
    rsis = rsis.sort_values(0,ascending=False)

    for i in range(3):
        add_text(1.04,0.55-i*0.024,rsis.iloc[i,1],rsis.iloc[i,0],bgcolor=color_map['line'][3-rsis.index[i]],color='white')


    kdjs = pd.DataFrame([[Jn[-1],Dn[-1],Kn[-1]],['J','D','K']]).T
    kdjs = kdjs.sort_values(0,ascending=False)

    for i in range(3):
        add_text(1.04,0.465-i*0.024,kdjs.iloc[i,1],kdjs.iloc[i,0],bgcolor=color_map['line'][3-kdjs.index[i]],color='white')


    if wr6[-1] > wr10[-1]:
        add_text(1.04,0.38-0*0.024,'WR6',wr6[-1],bgcolor=color_map['line'][-1],color='white')
        add_text(1.04,0.38-1.25*0.024,'WR10',wr10[-1],bgcolor=color_map['line'][-2],color='white')
    else:
        add_text(1.04,0.38-0*0.024,'WR6',wr6[-1],bgcolor=color_map['line'][-1],color='white')
        add_text(1.04,0.38-1.25*0.024,'WR10',wr10[-1],bgcolor=color_map['line'][-2],color='white')

    biass = pd.DataFrame([[bias24[-1],bias12[-1],bias6[-1]],['BIAS24','BIAS12','BIAS6']]).T
    biass = biass.sort_values(0,ascending=False)

    for i in range(3):
        add_text(1.04,0.29-i*0.023,biass.iloc[i,1],biass.iloc[i,0],bgcolor=color_map['line'][3-biass.index[i]],color='white')

    if OBV[-1]>maobv[-1]:
        add_text(1.07,0.2-0*0.024,'OBV',OBV[-1],bgcolor=color_map['line'][-3],color='white')
        add_text(1.07,0.2-1*0.024,'MAOBV',maobv[-1],bgcolor=color_map['line'][-1],color='white')
    else:
        add_text(1.07,0.2-1*0.024,'OBV',OBV[-1],bgcolor=color_map['line'][-3],color='white')
        add_text(1.07,0.2-0*0.024,'MAOBV',maobv[-1],bgcolor=color_map['line'][-1],color='white')

    add_text(1.04,0.1,'CCI',cci[-1],bgcolor=color_map['line'][0],color='white')
    add_text(1.04,0.02,'ROC',roc[-1],bgcolor=color_map['line'][-3],color='white')

    fig.update_xaxes(showline=True, linewidth=1, linecolor=color_map['line'][2], mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor=color_map['line'][2], mirror=True)

    fig.update_layout(showlegend=False,legend=dict(orientation = 'h',x=0.3,y=1.1),plot_bgcolor=color_map['bg'])

    return fig
