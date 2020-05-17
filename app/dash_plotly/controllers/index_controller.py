# Dash packages
import sqlite3

# Data manipulation packages
import pandas as pd
# Plotly packages
import plotly.express as px
from django_plotly_dash import DjangoDash

from dash_plotly.views.index_view import index_layout

con = sqlite3.connect("db.sqlite3")

app = DjangoDash('AsxIndexFragment')  # replaces dash.Dash

index_df = pd.read_sql_query('SELECT * FROM stock_asxindexdailyhistory order by index_date', con)
fig = px.line(index_df, x='index_date', y='close_index', color='index_name')
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

fig.update_layout(xaxis_rangeslider_visible=False,
                  xaxis_title="Date",
                  yaxis_title="Index",
                  )

app.layout = index_layout(fig)
