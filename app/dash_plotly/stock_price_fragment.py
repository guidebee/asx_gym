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

con = sqlite3.connect("db.sqlite3")

app = DjangoDash('StockPriceFragment')  # replaces dash.Dash

price_df = pd.read_sql_query('SELECT * FROM stock_stockpricedailyhistory where company_id=2 order by price_date', con)
fig = go.Figure([go.Scatter(x=price_df['price_date'], y=price_df['high_price'])])
fig.update_xaxes(
    rangeslider_visible=True,
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

app.layout = html.Div(children=[
    html.H1(children='Stock Price'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
