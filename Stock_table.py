from DatabaseManagement import *
import pandas as pd
import dash
import dash_table
from facts import securities
from dash_table.Format import Format, Scheme, Sign,Group, Prefix
def make_stock_table():
    df = get_data('holdings')
    df['Net'] = df['Price']*df['Amount']
    table = df[['Security','Amount','Net']].groupby('Security',as_index=False).sum().to_dict()

    prices, rate, profit, profit_rate  ={}, {},{},{}
    con = sql.connect('stock')
    index = 0
    for i in table['Security'].values():
        price = pd.read_sql_query("SELECT Close From %s"%i,con).values[-2:]
        prices[index] = price[-1].item()
        rate[index] = ((price[1]-price[0])/price[0]).item()
        profit[index] = (price[1]*table['Amount'][index] - table['Net'][index])[0]
        profit_rate[index] = (profit[index]/table['Net'][index]).item()
        index +=1
    con.close()

    table['Price'] = prices
    table['Rate'] = rate
    table['Profit'] = profit
    table['Return'] = profit_rate

    df = pd.DataFrame.from_dict(table)

    datatable = dash_table.DataTable(
        id='stock-table',
        columns=[
            {'name':'Name','id':'Security','type':'text','editable': False},
            {'name':'Hold','id':'Amount','type':'numeric','editable': False,
            'format':Format( group=True,precision=0, scheme=Scheme.fixed)},
            {'name':'Net','id':'Net','type':'numeric','editable': False,
            'format':Format(precision=4, scheme=Scheme.decimal_si_prefix)},
            {'name':'Price','id':'Price','type':'numeric','editable': False,
            'format':Format( group=True,precision=2, scheme=Scheme.fixed)},
            {'name':'Rate','id':'Rate','type':'numeric','editable': False,
            'format':Format(precision=2, scheme=Scheme.percentage,sign=Sign.positive)},
            {'name':'Profit','id':'Profit','type':'numeric','editable': False,
            'format':Format(precision=4, scheme=Scheme.decimal_si_prefix,sign=Sign.positive)},
            {'name':'Return','id':'Return','type':'numeric','editable': False,
            'format':Format(precision=2, scheme=Scheme.percentage,sign=Sign.positive)}
        ],
        data=df.to_dict('records'),
        sort_action="native",
        sort_mode="single",
        row_selectable="single",
        selected_rows = [0],

        style_table={
            'width': '100%',
            'overflowy': 'hidden','border':'none',
            'fontFamily':'Arial Black'
            },
        fixed_rows={'headers': True},
        style_header={
            'fontFamily':'Arial',
            'textAlign': 'right',
            'fontSize':14,
            'fontWeight':'bold',
            'whiteSpace': 'normal',
            'textOverflow': 'ellipsis',
            'height': 'auto',
            'backgroundColor':'#6155a6',
            'color':'#f4f4f2',
            'fontWeight': 'bold',
            'border':'none'
            },
        style_cell={
            'textAlign': 'right', 'border':'none','borderBottom':'2px solid #f4f4f2',
            'width':"auto",
            'fontFamily':'Arial','fontSize':16
        },
        style_as_list_view=True,
        style_cell_conditional=[
            {
            'if': {'column_id': 'Security'},
            'fontWeight':'bold',
            'textAlign':'left','fontFamily':'Arial','fontSize':18}
        ],
        style_header_conditional=[
            {
            'if': {'column_id': 'Security'},
            'textAlign':'left'}
        ],
        style_data_conditional = [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f4f4f2'
            },
            {'if': {
                'filter_query': '{Rate} < 0', # comparing columns to each other
                'column_id': ['Rate','Price']
                },
            'color': '#28abb9',
            'fontWeight':'bold'
            },
            {'if': {
                'filter_query': '{Rate} > 0', # comparing columns to each other
                'column_id': ['Rate','Price']
                },
            'color': '#c060a1',
            'fontWeight':'bold'
            },
            {'if': {
                'filter_query': '{Profit} < 0', # comparing columns to each other
                'column_id': ['Profit','Return']
                },
            'color': 'green',
            'fontWeight':'bold'
            },
            {'if': {
                'filter_query': '{Profit} > 0', # comparing columns to each other
                'column_id': ['Profit','Return']
                },
            'color': 'red',
            'fontWeight':'bold'
            },


        ]

        )


    return datatable, table
