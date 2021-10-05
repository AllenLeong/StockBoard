from StockBoard import app
from dash.dependencies import Input, Output,State
from DatabaseManagement import *
from Stock_table import make_stock_table
from Stock_chart import make_stock_chart
@app.callback(
    Output('url','pathname'),
    [Input('Navigation','value')]
)
def callbacks(value):
    return value


@app.callback(
    [Output('stock-table-page','children'),
    Output('table-memory','data')],
    [Input('url','pathname')]
)
def callbacks2(index):
    return make_stock_table()

@app.callback(
    Output('stock-chart','children'),
    [Input('table-memory','data'),
    Input('stock-table','selected_rows')]
)
def callbacks3(data,index):
    security = data['Security'][str(index[0])]
    return make_stock_chart(security)
