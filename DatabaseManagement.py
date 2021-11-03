from pandas_datareader import data
import yfinance as fy
import pandas as pd
from datetime import datetime
from facts import securities
import sqlite3 as sql
from tqdm import tqdm
import argparse
def YahooReader(security,start_date, end_date):

    df = fy.download(security, start_date, end_date)
    df['dif'] = df.Close.diff()
    df['Date'] = df.index
    df = df.fillna(0)
    return df

def make_stocktable(con,table_name):
    """
    Make a new table in database
    con:
        connection to database
    table_name:
        name of the table from the dictionary securities
        Note: Different from security name,use securities[security] to get the table name
    """
    make_table_sql = """
    CREATE TABLE %s (
        Open Double,
        High Double,
        Low Double,
        Close Double,
        Adj_Close Double,
        Volume Double,
        dif Double,
        Date TIMESTAMP PRIMARY KEY
    )
    """%table_name
    try:
        con.execute(make_table_sql)
    except:
        pass
    con.commit()

def drop_table(table_name):
    con = sql.connect('stock')
    try:
        con.execute('Drop table {}'.format(table_name))
    except:
        pass
    con.commit()
    con.close()

def insert_stocktable(df,con,table_name):
    """
    df:
        DataFrame of the stock price in the format of (Open,High,Low,Close,Adj_Close,Volume,dif,Date)
    con:
        connection to database
    table_name:
        table name in database related to the centain security
    """
    for i in range(len(df)):
        values = [table_name]+list(df.iloc[i,:-1])+["'"+str(df.iloc[i,-1])+"'"]
        try:
            sql_script = "INSERT INTO {}(Open,High,Low,Close,Adj_Close,Volume,dif,Date) VALUES ({},{},{},{},{},{},{},{})".format(*values)
            con.execute(sql_script)
        except:
            pass
        con.commit()


def update_data(security):
    """
    Input:
        security name
    Output:
        up-to-dated stock prices
    """
    con = sql.connect('stock')
    dates = pd.read_sql_query("SELECT Date from %s"%security, con)
    last_date = pd.Timestamp(dates.values[-1][0])
    today = pd.Timestamp(datetime.today().date())
    if today > last_date:
        new_data = YahooReader(securities[security],last_date,today)
        insert_stocktable(new_data.iloc[1:,],con,security)
        print(new_data)
    con.close()



def add_security(security):
    """
    add new security table into database
    Note: update the securities dictionary first.
    """
    today =  pd.Timestamp(datetime.today().date())
    df = YahooReader(securities[security],'2016-01-01',today)
    con = sql.connect('stock')
    #try:
    make_stocktable(con,security)
    insert_stocktable(df,con,security)
    #except:
    #    pass
    con.close()


def renew_security(security,start_date):
    """
    Some update would be wrong
    """
    today =  pd.Timestamp(datetime.today().date())
    df = YahooReader(security,start_date,today)
    con = sql.connect('stock')
    con.execute("DELETE FROM {} WHERE Date >= '{}'".format(security,start_date))
    insert_stocktable(df,con,security)
    con.commit()
    con.close()


def get_data(security):
    con = sql.connect('stock')
    df = pd.read_sql_query('SELECT * FROM %s'%security,con)
    con.close()
    return df

def record_operation(date, security,price, volume):
    con = sql.connect('stock')
    date = pd.Timestamp(date)
    values = (date,security,price,volume)
    sql_script = "INSERT INTO holdings (Date,Security,Price,Amount) VALUES ('{}','{}',{},{})".format(*values)
    con.execute(sql_script)
    con.commit()
    con.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage Database')
    parser.add_argument('-A', help = 'Update All Secrity', action = 'store_true')
    parser.add_argument('-a', help = 'Update A Secrity', type = str)
    parser.add_argument('-r', help = 'Add Record', action = 'store_true')
    parser.add_argument('--date', help = 'Add Record Date', type = str)
    parser.add_argument('--security', help = 'Add Record Security', type = str)
    parser.add_argument('--price', help = 'Add Record Price', type = float)
    parser.add_argument('--unit', help = 'Add Record Unit', type = float)
    parser.add_argument('-H', help = 'Check Holdings', action = 'store_true')
    parser.add_argument('-s', help = 'Check Secrity', type = str)

    args =  parser.parse_args()

    # 1. Update All
    if args.A:
        for i in tqdm(securities.keys()):
            update_data(i)
    ## 1.1 Update one
    if args.a:
        update_data(args.a)
    # 2. Add record
    if args.r:
        record_operation(args.date,args.security,args.price,args.unit)
    # 3. check holdings
    if args.H:
        print(get_data('holdings'))
    # 4. check Secrity
    if args.s:
        print(get_data(args.s))

# con = sql.connect('stock')
# drop_table('holdings')
# con.execute('CREATE TABLE holdings(Date TIMESTAMP, Security Varchar(255), Price Double, Amount Double) ')
# con.commit()
# con.close()
#add_security('AAPL')
# con = sql.connect('stock')
# con.execute("Delete from holdings where Security= 'SSEC'")
# df = pd.read_sql_query("SELECT * from holdings", con)
# print(df)
# con.commit()
# con.close()
# for i in tqdm(securities.keys()):
#    drop_table(i)
#    print('successfully drop table %s'%i)
#    add_security(i)
#    print('successfully add %s'%i)
#    print(get_data(i))
#record_operation('2020-02-13','AAPL',135,10)
# record_operation('2020-02-13','DJI',0,0)
# record_operation('2021-01-15','SZC',0,0)
# record_operation('2021-01-12','SSEC',0,0)

# print(get_data('holdings'))
# for i in tqdm(securities.keys()):
#    update_data(i)
#
#print(get_data("AAPL"))
