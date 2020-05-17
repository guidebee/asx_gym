from django.apps import AppConfig

import dash_plotly.controllers.index_controller
import dash_plotly.controllers.price_controller


class DashPlotlyConfig(AppConfig):
    name = 'dash_plotly'
