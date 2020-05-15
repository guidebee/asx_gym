from django.urls import path

from .views import IndexView, PriceView

app_name = 'main'

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('price', PriceView.as_view(), name='price'),
]
