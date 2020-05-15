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

app = DjangoDash('BootstrapApplication')  # replaces dash.Dash

index_df = pd.read_sql_query('SELECT * FROM stock_asxindexdailyhistory', con)
fig = px.line(index_df, x='index_date', y='open_index', color='index_name')

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
