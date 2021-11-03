import pandas as pd
import numpy as np

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

def kdj(dff):
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
    return Kn,Dn,Jn

def wr(dff,n):
    Hn = dff.High.rolling(n).max()
    Hn.fillna(value=dff.High.expanding().max(), inplace=True)
    Ln = dff.Low.rolling(n).min()
    Ln.fillna(value=dff.Low.expanding().min(), inplace=True)
    Cn = dff.Close

    return (Hn-Cn)/(Hn-Ln)*100

def obv_volume(x):
    if x>0: return 1
    elif x==0: return 0
    else: return -1
def obv(dff):
    obv = [0]
    for i in dff.dif.apply(obv_volume)*dff.Volume:
        obv.append(obv[-1]+i)
    return pd.Series(data=np.array(obv[1:]),index=dff.index)


def cci(dff,n=20):
    TP = (dff.High + dff.Low+ dff.Close)/3

    MA = TP.rolling(n).mean()
    MD = (MA-dff.Close).rolling(n).std()
    cci = (TP-MA)/(MD*0.015)
    return cci

def ROC(dff,n):
    return (dff.Close - dff.Close.shift(n))/dff.Close.shift(n)*100
