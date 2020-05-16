# Dash packages
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Plotly packages
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Data manipulation packages
import numpy as np
import pandas as pd
import sqlite3

from dash.exceptions import PreventUpdate

con = sqlite3.connect("db.sqlite3")

app = DjangoDash('StockPriceFragment', add_bootstrap_links=True)  # replaces dash.Dash

company_df = pd.read_sql_query('SELECT id,name,description,code,sector_id FROM stock_company', con)
sector_df = pd.read_sql_query('SELECT id,name,full_name FROM stock_sector', con)


def get_company_stock_data(value):
    con = sqlite3.connect("db.sqlite3")

    company_id = int(value)

    company = company_df[company_df['id'] == company_id]

    company_name = company.iloc[0, 1]
    company_desc = company.iloc[0, 2]
    sector_id = company.iloc[0, 4]
    sector_info = sector_df[sector_df['id'] == sector_id].iloc[0, 2]
    opacity = 1.0
    price_df = pd.read_sql_query(
        f'SELECT * FROM stock_stockpricedailyhistory where company_id={company_id} order by price_date',
        con)
    fig = go.Figure(
        [
            go.Scatter(x=price_df['price_date'], y=price_df['high_price'], name="High", opacity=opacity),
            go.Scatter(x=price_df['price_date'], y=price_df['low_price'], name="Low", opacity=opacity),
            go.Scatter(x=price_df['price_date'], y=price_df['open_price'], name="Open", opacity=opacity),
            go.Scatter(x=price_df['price_date'], y=price_df['close_price'], name="Close", opacity=opacity),
            go.Candlestick(x=price_df['price_date'],
                           open=price_df['open_price'],
                           high=price_df['high_price'],
                           low=price_df['low_price'],
                           close=price_df['close_price'], name="Candle Stick")
        ],

    )

    fig.update_layout(xaxis_rangeslider_visible=True,
                      title=company_name,
                      xaxis_title="Date",
                      yaxis_title="Price",
                      )

    fig.update_xaxes(

        rangeselector=dict(

            buttons=list([
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),

                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),

    )

    con.close()
    return fig, company_desc, f'Sector:{sector_info}'


fig, company_desc, sector_info = get_company_stock_data(2)

app.layout = dbc.Container(children=[
    dbc.Row(
        [
            dbc.Col(dcc.Dropdown(id='company_name', placeholder='company name'), width=12),
        ]
    ),

    dcc.Graph(
        id='stock-price-graph',
        figure=fig,

    ),
    dbc.Row(
        [
            dbc.Col(html.Div(id='company-info', children=company_desc)),

        ]
    ),
    dbc.Row(
        [
            dbc.Col(html.Div(id='sector-info', children=sector_info)),

        ]
    ),
])


@app.callback(
    Output("company_name", "options"),
    [Input("company_name", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate

    options = []
    df = company_df[company_df['name'].str.contains(search_value, case=False)]

    for i in range(len(df)):
        options.append({"label": df.iloc[i, 1], "value": df.iloc[i, 0]})

    return options


@app.callback(
    [
        Output('stock-price-graph', 'figure'),
        Output('company-info', 'children'),
        Output('sector-info', 'children')
    ],
    [Input('company_name', 'value')])
def update_stock_price(value):
    if not value:
        raise PreventUpdate
    try:
        return get_company_stock_data(value)

    except Exception as e:
        print(e)
        raise PreventUpdate
