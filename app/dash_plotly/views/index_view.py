import dash_core_components as dcc
import dash_html_components as html


def index_layout(fig):
    return html.Div(children=[
        html.H3(children='ASX Indexes'),

        dcc.Graph(
            id='example-graph',
            figure=fig
        )
    ])
