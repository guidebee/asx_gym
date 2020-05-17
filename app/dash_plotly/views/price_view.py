import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def price_layout(fig, company_desc, sector_info):
    return dbc.Container(children=[
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
